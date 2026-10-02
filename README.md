# prompt-cache-meter

## What this is

OpenAI caches the start of your prompt. When the next request starts with the same tokens, those tokens are cheaper and faster. This project shows how much a given conversation pattern would save, and why.

For every request it prints two things side by side:

- **What OpenAI would bill.** Our own simulator works this out from OpenAI's caching rules: tokens served from cache, tokens written to cache, tokens at full price, and the cost.
- **What a local model actually did.** The same request goes to Ollama, and we record its token counts, cached tokens and response time.

## How it works

1. A **scenario** builds a sequence of requests. Examples: the same prompt twice, a 60-turn chat, or a RAG question an hour later.
2. Each request is **counted** with OpenAI's tokenizer (tiktoken).
3. The **simulator** checks whether the start of the prompt matches anything still in its cache, then works out what is read, what is written and what is billed at full price.
4. Unless `--dry-run` is set, the request also goes to **Ollama**.
5. The results are printed as tables and saved to `runs/`.

The simulator uses a fake clock, so a scenario can jump ahead an hour to test cache expiry without waiting.

All the rules and numbers (minimum prompt size, TTLs, write and read prices, model prices) are in [`src/prompt_cache_meter/cache_rules.toml`](src/prompt_cache_meter/cache_rules.toml). That file notes which values come from OpenAI's docs and which are assumptions, so you can correct them there. For the architecture, see [`DESIGN.md`](DESIGN.md).

## Setup

```sh
cp .env.example .env   # set OLLAMA__DEFAULT_MODEL (and OPENROUTER__API_KEY to use OpenRouter)
uv sync
uv run pre-commit install
```

## Run

```sh
uv run prompt-cache-meter scenarios --dry-run                      # all scenarios, simulator only
uv run prompt-cache-meter scenarios --dry-run --generation legacy  # older OpenAI models' rules
uv run prompt-cache-meter scenarios cold-warm-broken               # one scenario, also calls Ollama
uv run prompt-cache-meter scenarios long-chat --summary-only       # only the summary table
uv run prompt-cache-meter scenarios --model llama3.1:8b            # another Ollama model
uv run prompt-cache-meter scenarios --rules my_rules.toml          # your own corrected rules and prices
```

Scenarios: `cold-warm-broken`, `question-wording`, `name-placement`, `long-chat`, `rag-ttl`, `tenants`, `short-prompt`.

Some tips:

- Ollama silently cuts prompts longer than its context window. The tool warns you when a prompt gets close. For the long chat, start Ollama with a bigger window: `OLLAMA_CONTEXT_LENGTH=16384 ollama serve`.
- Ollama keeps its own cache between scenarios, so it can show cache hits the simulator, which starts empty for every scenario, does not.

To check that a provider is reachable before a long run, send it one prompt. It prints the reply, token usage (including cached tokens) and latency:

```sh
uv run prompt-cache-meter ping                          # Ollama
uv run prompt-cache-meter ping --provider openrouter    # OpenRouter, needs OPENROUTER__API_KEY
```

## Proxy: simulate any app's calls

The proxy lets you see the simulation for real traffic, not just the built-in scenarios. It is a small local server that speaks the OpenAI Chat Completions API. Point any app at it, and it forwards every call unchanged to the real provider, simulates what OpenAI would bill, and logs both.

```sh
uv run prompt-cache-meter proxy                                         # forwards to Ollama on port 8787
uv run prompt-cache-meter proxy --upstream https://api.openai.com/v1    # forwards to OpenAI
```

Then change only the base URL in your app:

```python
client = OpenAI(base_url="http://127.0.0.1:8787/v1", api_key="...")
```

Each call prints one line (simulated cached / written / full-price tokens and cost, next to the provider's own usage) and is saved to `runs/proxy/`. Message content is never logged. If the simulator cannot read a request, the call still goes through and the line shows a warning.

Models that are not in the rules file are priced as `--price-as` (by default, the generation's default model). The simulated cache lives in memory and starts empty each time the proxy starts.

For setup with other SDKs and frameworks, reading the output and troubleshooting, see [`PROXY_MANUAL.md`](PROXY_MANUAL.md).

## Test

```sh
uv run pytest                                  # unit tests, no network needed
uv run pytest --integration                    # also call a real Ollama
uv run ruff format && uv run ruff check --fix
uv run mypy
```

The pre-commit hook runs format, lint, mypy and the unit tests on every commit.
