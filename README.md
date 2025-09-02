
https://github.com/user-attachments/assets/b5094911-21e3-4867-838d-03275eb6af0e


TokenTrie Structure:
```plain
        (root)
       /    |    \
      p     c     ...
     /      |
    r       l
   /        a
  i         s
 /|         s
n t         |
tස        (print, ID:123) (private, ID:456) (class, ID:789)

DFA-Like Prefix Matching for "pri":
State 0: prefix="pri"
  ├── Allowed tokens: {print:123, private:456}
  ├── Transition: Select "print" (deterministic, only one token starts with "pri")
State 1: prefix="" (fully matched)
  ├── Allowed tokens: {(, ID:101), [, ID:102), etc.}
  ├── Transition: Sample probabilistically, e.g., select "("
State 2: Continue sampling until max_new_tokens or newline
```

## Features
- **PCAS Backend**: Leverages a Hugging Face transformer model with a DFA-inspired, trie-based algorithm to constrain token sampling, ensuring exact prefix adherence.
- **TokenTrie**: A custom trie structure to efficiently manage token vocabularies and enforce prefix constraints.
- **DFA-Like Prefix Enforcement**: Models prefix matching as a state machine, guaranteeing valid token sequences.
- **Streaming Support**: Streams generated tokens in real-time via Server-Sent Events (SSE).
- **Configurable Parameters**: Supports `temperature`, `top_p`, `top_k`, and `max_new_tokens` for generation control.
- **Language-Agnostic**: Handles various programming languages with optional language specification.
- **Code-Only Mode**: Outputs pure code without explanations, ideal for editor integration.
- **Ollama Backend**: Included for side-by-side benchmarking, but less precise than PCAS.
- **CORS Support**: Enables cross-origin requests for web-based integrations.

## How It Works

### Key Components

1. **Settings**: Configures the service using environment variables.
   - `CPC_BACKEND`: Defaults to `pcas` (primary) or `ollama` (benchmarking).
   - `HF_MODEL`: Hugging Face model for PCAS (default: `HuggingFaceTB/SmolLM-135M`).
   - `OLLAMA_URL`: Ollama API endpoint (default: `http://localhost:11434`).
   - `OLLAMA_MODEL`: Ollama model name (default: `gpt-oss:20b`).

2. **TokenTrie**:
   - A trie-based data structure that maps the model's token vocabulary to strings.
   - Supports two key queries:
     - `tokens_that_start_with(s)`: Returns tokens that start with a given string.
     - `tokens_that_are_prefix_of(s)`: Returns tokens that are prefixes of a given string.
   - Enables efficient prefix constraint enforcement by identifying valid tokens for each generation step.
   - Example: For prefix `pri`, the trie identifies tokens like `print` or `private` as valid continuations.

3. **PCAS (Prefix-Constrained Autoregressive Sampling)**:
   - Combines a Hugging Face transformer model with a DFA-like approach to enforce prefix constraints.
   - Workflow:
     1. Encodes the input context using the model's tokenizer.
     2. Initializes a `TokenTrie` with the model's vocabulary.
     3. Uses the trie to compute allowed tokens at each step, ensuring they match or extend the prefix.
     4. Samples tokens autoregressively, constrained to the allowed set, using parameters like `temperature`, `top_p`, and `top_k`.
     5. Forces single-token selections when only one valid token matches the prefix, mimicking a DFA's deterministic transitions.
     6. Continues generation up to `max_new_tokens` or until a newline is encountered.
   - The DFA-like behavior ensures the generated sequence respects the prefix by modeling valid token transitions as states.

4. **Ollama Backend** (Benchmarking Only):
   - Used for side-by-side comparison with PCAS.
   - Constructs a prompt with context and prefix, sent to an external Ollama API.
   - Relies on prompt engineering to approximate prefix adherence, less precise than PCAS.
   - Supports streaming and non-streaming modes but is primarily for evaluating PCAS performance.

5. **API Endpoints**:
   - `/cpc`: Returns a `CPCResponse` with the generated suggestion, call count, backend used, and debug logs.
   - `/cpc_stream`: Streams tokens via SSE, including metadata, time-to-first-byte (TTFB), and completion events.

6. **Request/Response Models**:
   - `CPCRequest`: Accepts `context`, `prefix`, `max_new_tokens`, `temperature`, `top_p`, `top_k`, `mode`, `lang`, and `code_only`.
   - `CPCResponse`: Returns `suggestion`, `calls_prefix` (model calls for prefix), `backend`, and `debug` logs.

### Workflow
1. **Input**: Client sends a `CPCRequest` with code context (previous code) and prefix (partial code to continue).
2. **Backend Selection**: Defaults to PCAS unless `ollama` is specified for benchmarking.
3. **PCAS Generation**:
   - Tokenizes the context and initializes the `TokenTrie`.
   - Iteratively samples tokens, constrained by the trie to match the prefix.
   - Uses a DFA-like approach to enforce deterministic transitions when only one token is valid.
   - Outputs the suggestion, either as a single response or streamed via SSE.
4. **Ollama Generation** (Benchmarking):
   - Builds a prompt with context and prefix, sent to the Ollama API.
   - Collects or streams the response, stopping at delimiters like newlines.
5. **Output**: Returns a `CPCResponse` or streams events, ensuring the prefix is respected.

### Solving the Character Prefix Conditioning Problem
The character prefix conditioning problem requires generated text to start exactly with a given prefix, which is challenging for autoregressive models that generate token-by-token. PCAS solves this using a DFA-inspired approach:

- **TokenTrie**: Efficiently identifies tokens that match or extend the prefix, reducing the search space.
- **DFA-Like Sampling**:
  - Models prefix matching as a state machine, where each state represents the current position in the prefix.
  - Transitions are constrained to tokens that are valid continuations, as determined by the trie.
  - When only one token is valid (e.g., `print` for prefix `pri`), it is selected deterministically, mimicking a DFA.
  - For multiple valid tokens, probabilistic sampling (with `temperature`, `top_p`, `top_k`) selects the next token.
- **Example**: For prefix `pri` in context `def hello_world():\n    `, the trie ensures tokens like `print` are prioritized, while invalid tokens (e.g., `class`) are excluded.
- **Precision**: Unlike prompt-based methods, PCAS guarantees prefix adherence by constraining the token logits, avoiding deviations.

The Ollama backend, used for benchmarking, relies on prompt engineering, which is less reliable and included only to compare performance against PCAS's deterministic approach.

## Installation
1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   Required packages: `fastapi`, `httpx`, `torch`, `transformers`, `pydantic`, `uvicorn`.

3. **Set Environment Variables** (optional):
   ```bash
   export CPC_BACKEND="pcas"
   export HF_MODEL="HuggingFaceTB/SmolLM-135M"
   export OLLAMA_URL="http://localhost:11434"
   export OLLAMA_MODEL="gpt-oss:20b"
   ```

4. **Run the Service**:
   ```bash
   python main.py
   ```
   The service runs on `http://0.0.0.0:6969`.

## Usage
### Non-Streaming Request
Send a POST request to `/cpc`:
```json
{
  "context": "def hello_world():\n    ",
  "prefix": "pri",
  "max_new_tokens": 40,
  "temperature": 0.5,
  "top_p": 0.92,
  "top_k": 50,
  "code_only": true
}
```
**Response**:
```json
{
  "suggestion": "print('Hello, World!')",
  "calls_prefix": 2,
  "backend": "pcas",
  "debug": ["[forced] 'print'", "[branch] |A|=5", "  sampled '('"]
}
```

### Streaming Request
Send a POST request to `/cpc_stream`:
```
event: meta
data: {"backend": "pcas"}

event: ttfb
data: {"ms": 123}

event: chunk
data: {"text": "print"}

event: chunk
data: {"text": "("}

event: done
data: {"calls_prefix": 2, "tokens_emitted": 5, "ms_total": 456}
```

### Query Parameter
Override the backend with `backend=ollama` for benchmarking (e.g., `/cpc?backend=ollama`).

## Example
**Context**:
```python
def hello_world():
    pri
```
**Prefix**: `pri`
**Request**:
```json
{
  "context": "def hello_world():\n    ",
  "prefix": "pri",
  "code_only": true
}
```
**PCAS Output**:
```json
{
  "suggestion": "print('Hello, World!')",
  "calls_prefix": 2,
  "backend": "pcas"
}
```
