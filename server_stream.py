from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Set, Tuple

import httpx
import torch
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel


@dataclass(frozen=True)
class Settings:
    backend: str = os.environ.get("CPC_BACKEND", "pcas")
    hf_model: str = os.environ.get("HF_MODEL", "HuggingFaceTB/SmolLM-135M")
    ollama_url: str = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    ollama_model: str = os.environ.get("OLLAMA_MODEL", "gpt-oss:20b")


SETTINGS = Settings()


class CPCRequest(BaseModel):
    context: str
    prefix: str
    max_new_tokens: int = 40
    temperature: float = 0.5
    top_p: float = 0.92
    top_k: int = 50
    mode: Optional[str] = None  
    lang: Optional[str] = None  
    code_only: bool = True      


class CPCResponse(BaseModel):
    suggestion: str
    calls_prefix: int
    backend: str
    debug: Optional[List[str]] = None



class TokenTrie:
    def __init__(self, vocab_texts: Sequence[str]):
        self.child: List[dict[str, int]] = [{}]
        self.ends: List[List[int]] = [[]]
        self.all: List[List[int]] = [[]]
        for tid, s in enumerate(vocab_texts):
            self._add(tid, s)
        self._fill_all()

    def _new(self) -> int:
        self.child.append({})
        self.ends.append([])
        self.all.append([])
        return len(self.child) - 1

    def _add(self, tid: int, s: str) -> None:
        n = 0
        for ch in s:
            n = self.child[n].setdefault(ch, self._new())
        self.ends[n].append(tid)

    def _fill_all(self) -> None:
        for n in reversed(range(len(self.child))):
            acc: List[int] = list(self.ends[n])
            for nxt in self.child[n].values():
                acc.extend(self.all[nxt])
            self.all[n] = acc

    def node_for_prefix(self, s: str) -> Optional[int]:
        n = 0
        for ch in s:
            if ch not in self.child[n]:
                return None
            n = self.child[n][ch]
        return n

    def tokens_that_start_with(self, s: str) -> Set[int]:
        n = self.node_for_prefix(s)
        return set(self.all[n]) if n is not None else set()

    def tokens_that_are_prefix_of(self, s: str) -> Set[int]:
        n = 0
        out: List[int] = []
        for ch in s:
            if self.ends[n]:
                out.extend(self.ends[n])
            if ch not in self.child[n]:
                break
            n = self.child[n][ch]
        else:
            if self.ends[n]:
                out.extend(self.ends[n])
        return set(out)


class PCAS:
    def __init__(self, prefix: str, vocab_texts: Sequence[str], trie: TokenTrie):
        self._prefix = prefix
        self._i = 0
        self._vocab = vocab_texts
        self._trie = trie

    def done(self) -> bool:
        return self._i >= len(self._prefix)

    def allowed(self) -> Optional[Set[int]]:
        if self.done():
            return None
        remainder = self._prefix[self._i :]
        a = self._trie.tokens_that_are_prefix_of(remainder)
        b = self._trie.tokens_that_start_with(remainder)
        return a | b

    def advance(self, tid: int) -> None:
        if self.done():
            return
        tok = self._vocab[tid]
        remainder = self._prefix[self._i :]
        if remainder.startswith(tok):
            self._i += len(tok)
        elif tok.startswith(remainder):
            self._i = len(self._prefix)
        else:
            raise ValueError("Token incompatible with prefix")


def sample_from_logits(
    logits: torch.Tensor,
    *,
    temperature: float = 0.5,
    allowed_ids: Optional[Set[int]] = None,
    top_p: float = 0.92,
    top_k: int = 50,
    rng: Optional[random.Random] = None,
) -> int:
    if rng is None:
        rng = random
    x = logits.clone().squeeze(0)

    if allowed_ids is not None:
        mask = torch.full_like(x, float("-inf"))
        idx = list(allowed_ids)
        if idx:
            mask[idx] = 0.0
        x = x + mask

    x = x / max(1e-6, temperature)

    if top_k and top_k > 0:
        vals, ids = torch.topk(x, k=min(top_k, x.numel()))
        z = torch.full_like(x, float("-inf"))
        z[ids] = vals
        x = z

    if 0.0 < top_p < 1.0:
        s, si = torch.sort(x, descending=True)
        p = torch.softmax(s, dim=-1)
        c = torch.cumsum(p, dim=-1)
        cut = torch.nonzero(c > top_p, as_tuple=False)
        if cut.numel():
            s[cut[0, 0] + 1 :] = float("-inf")
        z = torch.full_like(x, float("-inf"))
        z[si] = s
        x = z

    probs = torch.softmax(x, dim=-1).detach().cpu().numpy()
    return rng.choices(range(len(probs)), weights=probs, k=1)[0]

_HF_READY = False
_device: str
_tokenizer = None
_model = None
_vocab_texts: List[str]
_trie: TokenTrie


def _ensure_hf() -> None:
    global _HF_READY, _device, _tokenizer, _model, _vocab_texts, _trie
    if _HF_READY:
        return
    from transformers import AutoModelForCausalLM, AutoTokenizer  # local import

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    _tokenizer = AutoTokenizer.from_pretrained(SETTINGS.hf_model)
    _model = AutoModelForCausalLM.from_pretrained(SETTINGS.hf_model).to(_device).eval()
    _vocab_texts = [_tokenizer.decode([i]) for i in range(_tokenizer.vocab_size)]
    _trie = TokenTrie(_vocab_texts)
    _HF_READY = True

def _pcas_generate(req: CPCRequest) -> CPCResponse:
    _ensure_hf()
    rng = random.Random(0)

    ctx_ids = _tokenizer.encode(req.context)
    if not ctx_ids:
        bos = _tokenizer.bos_token_id or _tokenizer.eos_token_id
        ctx_ids = [bos]

    x = torch.tensor([ctx_ids], device=_device)
    out = _model(x, use_cache=True)
    past = out.past_key_values
    last = out.logits[:, -1, :]

    pcas = PCAS(req.prefix, _vocab_texts, _trie)
    generated: List[int] = []
    debug: List[str] = []
    calls = 0

    forced: List[int] = []
    while not pcas.done():
        allowed = pcas.allowed()
        if not allowed:
            return CPCResponse(suggestion="", calls_prefix=calls, backend="pcas", debug=debug + ["A(i)=∅"])
        if len(allowed) == 1:
            t = next(iter(allowed))
            forced.append(t)
            pcas.advance(t)
            debug.append(f"[forced] '{_vocab_texts[t]}'")
        else:
            break

    if forced:
        xx = torch.tensor([forced], device=_device)
        out = _model(xx, use_cache=True, past_key_values=past)
        past = out.past_key_values
        last = out.logits[:, -1, :]
        generated.extend(forced)

    while not pcas.done():
        allowed = pcas.allowed()
        debug.append(f"[branch] |A|={len(allowed)}")
        t = sample_from_logits(
            last,
            temperature=req.temperature,
            allowed_ids=allowed,
            top_p=req.top_p,
            top_k=req.top_k,
            rng=rng,
        )
        debug.append(f"  sampled '{_vocab_texts[t]}'")
        generated.append(t)
        pcas.advance(t)

        xx = torch.tensor([[t]], device=_device)
        out = _model(xx, use_cache=True, past_key_values=past)
        past = out.past_key_values
        last = out.logits[:, -1, :]
        calls += 1

        forced = []
        while not pcas.done():
            allowed = pcas.allowed()
            if len(allowed) == 1:
                tf = next(iter(allowed))
                forced.append(tf)
                pcas.advance(tf)
                generated.append(tf)
                debug.append(f"[forced] '{_vocab_texts[tf]}'")
            else:
                break
        if forced:
            xx = torch.tensor([forced], device=_device)
            out = _model(xx, use_cache=True, past_key_values=past)
            past = out.past_key_values
            last = out.logits[:, -1, :]

    for _ in range(req.max_new_tokens):
        t = sample_from_logits(
            last,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            rng=rng,
        )
        generated.append(t)
        xx = torch.tensor([[t]], device=_device)
        out = _model(xx, use_cache=True, past_key_values=past)
        past = out.past_key_values
        last = out.logits[:, -1, :]
        if _vocab_texts[t].endswith("\n"):
            break

    return CPCResponse(
        suggestion=_tokenizer.decode(generated),
        calls_prefix=calls,
        backend="pcas",
        debug=debug,
    )



def _build_code_prompt(context: str, prefix: str, lang: Optional[str], code_only: bool) -> str:
    lang_name = lang or "code"
    if code_only:
        instr = "You are a code autocomplete engine. "
        rules = (
            "- Output ONLY code (no explanations, no backticks).\n"
            "- Continue from exactly after the <PREFIX>. Do not repeat <CONTEXT> or <PREFIX>.\n"
            "- Maintain indentation and style from <CONTEXT>.\n"
        )
    else:
        instr = "You are an assistant. "
        rules = "- Continue the text from <PREFIX>.\n"

    return (
        f"{instr}Continue the user's {lang_name} source file.\n"
        f"{rules}\n"
        "<CONTEXT>\n"
        f"{context}\n"
        "</CONTEXT>\n"
        "<PREFIX>\n"
        f"{prefix}\n"
        "</PREFIX>\n"
        "Completion:\n"
    )


async def _ollama_generate(req: CPCRequest) -> CPCResponse:
    prompt = _build_code_prompt(req.context, req.prefix, req.lang, req.code_only)
    options = {
        "temperature": req.temperature,
        "top_p": req.top_p,
        "top_k": req.top_k,
        "num_predict": req.max_new_tokens + 64,
        "stop": ["\n\n", "\r\n\r\n", "</CONTEXT>", "</PREFIX>", "```"]
    }
    body = {"model": SETTINGS.ollama_model, "prompt": prompt, "options": options, "stream": True}

    parts: List[str] = []
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(f"{SETTINGS.ollama_url}/api/generate", json=body)
        r.raise_for_status()
        async for line in r.aiter_lines():
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "response" in obj:
                chunk = obj["response"]
                parts.append(chunk)
                if "\n" in chunk:
                    break
            if obj.get("done"):
                break

    suggestion = req.prefix + "".join(parts)
    return CPCResponse(suggestion=suggestion, calls_prefix=1, backend="ollama", debug=["ollama prompt-only code mode"])



def _sse(event: str, data: dict) -> str:
    return f"event: {event}\n" f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _pcas_stream(req: CPCRequest):
    _ensure_hf()

    t0 = time.perf_counter()
    rng = random.Random(0)

    ctx_ids = _tokenizer.encode(req.context)
    if not ctx_ids:
        bos = _tokenizer.bos_token_id or _tokenizer.eos_token_id
        ctx_ids = [bos]

    x = torch.tensor([ctx_ids], device=_device)
    out = _model(x, use_cache=True)
    past = out.past_key_values
    last = out.logits[:, -1, :]

    pcas = PCAS(req.prefix, _vocab_texts, _trie)
    generated: List[int] = []
    calls = 0
    ttfb_emitted = False

    yield _sse("meta", {"backend": "pcas"})

    forced: List[int] = []
    while not pcas.done():
        allowed = pcas.allowed()
        if not allowed:
            yield _sse("done", {"calls_prefix": calls, "error": "A(i)=∅"})
            return
        if len(allowed) == 1:
            t = next(iter(allowed))
            forced.append(t)
            pcas.advance(t)
        else:
            break

    if forced:
        xx = torch.tensor([forced], device=_device)
        out = _model(xx, use_cache=True, past_key_values=past)
        past = out.past_key_values
        last = out.logits[:, -1, :]
        for t in forced:
            piece = _vocab_texts[t]
            if not ttfb_emitted:
                yield _sse("ttfb", {"ms": int((time.perf_counter() - t0) * 1000)})
                ttfb_emitted = True
            generated.append(t)
            yield _sse("chunk", {"text": piece})

    while not pcas.done():
        allowed = pcas.allowed()
        t = sample_from_logits(
            last,
            temperature=req.temperature,
            allowed_ids=allowed,
            top_p=req.top_p,
            top_k=req.top_k,
            rng=rng,
        )
        piece = _vocab_texts[t]
        if not ttfb_emitted:
            yield _sse("ttfb", {"ms": int((time.perf_counter() - t0) * 1000)})
            ttfb_emitted = True
        generated.append(t)
        yield _sse("chunk", {"text": piece})
        pcas.advance(t)

        xx = torch.tensor([[t]], device=_device)
        out = _model(xx, use_cache=True, past_key_values=past)
        past = out.past_key_values
        last = out.logits[:, -1, :]
        calls += 1

        forced = []
        while not pcas.done():
            allowed = pcas.allowed()
            if len(allowed) == 1:
                tf = next(iter(allowed))
                forced.append(tf)
                pcas.advance(tf)
                generated.append(tf)
            else:
                break
        if forced:
            xx = torch.tensor([forced], device=_device)
            out = _model(xx, use_cache=True, past_key_values=past)
            past = out.past_key_values
            last = out.logits[:, -1, :]
            for tf in forced:
                yield _sse("chunk", {"text": _vocab_texts[tf]})

    for _ in range(req.max_new_tokens):
        t = sample_from_logits(
            last,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            rng=rng,
        )
        generated.append(t)
        xx = torch.tensor([[t]], device=_device)
        out = _model(xx, use_cache=True, past_key_values=past)
        past = out.past_key_values
        last = out.logits[:, -1, :]
        yield _sse("chunk", {"text": _vocab_texts[t]})
        if _vocab_texts[t].endswith("\n"):
            break

    yield _sse(
        "done",
        {"calls_prefix": calls, "tokens_emitted": len(generated), "ms_total": int((time.perf_counter() - t0) * 1000)},
    )


async def _ollama_stream(req: CPCRequest):
    t0 = time.perf_counter()
    yield _sse("meta", {"backend": "ollama"})

    prompt = _build_code_prompt(req.context, req.prefix, req.lang, req.code_only)
    options = {
        "temperature": req.temperature,
        "top_p": req.top_p,
        "top_k": req.top_k,
        "num_predict": req.max_new_tokens + 64,
        "stop": ["\n\n", "\r\n\r\n", "</CONTEXT>", "</PREFIX>", "```"]
    }
    body = {"model": SETTINGS.ollama_model, "prompt": prompt, "options": options, "stream": True}

    first = True
    emitted = 0
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(f"{SETTINGS.ollama_url}/api/generate", json=body)
        r.raise_for_status()
        async for line in r.aiter_lines():
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            if "response" in obj:
                if first:
                    yield _sse("ttfb", {"ms": int((time.perf_counter() - t0) * 1000)})
                    first = False
                chunk = obj["response"]
                emitted += len(chunk)
                yield _sse("chunk", {"text": chunk})
            if obj.get("done"):
                break

    yield _sse("done", {"bytes": emitted, "ms_total": int((time.perf_counter() - t0) * 1000)})



app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


@app.post("/cpc", response_model=CPCResponse)
async def cpc(req: CPCRequest, backend: Optional[str] = Query(default=None)):
    mode = backend or req.mode or SETTINGS.backend
    if mode == "pcas":
        return _pcas_generate(req)
    return await _ollama_generate(req)


@app.post("/cpc_stream")
async def cpc_stream(req: CPCRequest, backend: Optional[str] = Query(default=None)):
    mode = backend or req.mode or SETTINGS.backend

    async def gen():
        if mode == "pcas":
            async for evt in _pcas_stream(req):
                yield evt
        else:
            async for evt in _ollama_stream(req):
                yield evt

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(gen(), media_type="text/event-stream", headers=headers)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=6969)

