# Proxy manual: simulate OpenAI cache billing for any app

The proxy lets you see what OpenAI's prompt caching would do to **your own app's** LLM calls, without changing the app's code. You start the proxy and point the app's base URL at it. The app keeps working as before, and every chat call gets a line in the console with the simulated cache bill.

```
your app  ->  proxy (127.0.0.1:8787)  ->  Ollama / OpenRouter / OpenAI
                  |
                  +-> console line + runs/proxy/<date>.jsonl
```

The proxy forwards every request and response byte for byte. It never changes what your app sends or receives.

## 1. Start the proxy

From this project's folder:

```sh
uv run prompt-cache-meter proxy
```

You should see:

```
Proxy on http://127.0.0.1:8787/v1 -> http://127.0.0.1:11434/v1 (new rules)
```

Leave it running in its own terminal. Stop it with `Ctrl+C`.

### Options

| Option | Default | Use it to |
| --- | --- | --- |
| `--upstream URL` | `http://127.0.0.1:11434/v1` (Ollama) | Pick the real provider. Any OpenAI-compatible base URL, ending in `/v1`. |
| `--generation new\|legacy` | `new` | Pick which OpenAI rules to simulate. `new` = GPT-5.6 and later, `legacy` = earlier models. |
| `--price-as MODEL` | `gpt-6.1-terra` (new) / `gpt-5` (legacy) | Choose the OpenAI price for models not in the rules file, such as `qwen2.5:14b`. |
| `--port N` | `8787` | Run on another port, for example to run two proxies at once. |
| `--host ADDR` | `127.0.0.1` | Listen on another interface. See [Docker and other machines](#docker-and-other-machines). |
| `--rules FILE` | `src/prompt_cache_meter/cache_rules.toml` | Use your own corrected rules and prices. |

### Common setups

```sh
uv run prompt-cache-meter proxy                                                  # local Ollama
uv run prompt-cache-meter proxy --upstream https://openrouter.ai/api/v1          # OpenRouter
uv run prompt-cache-meter proxy --upstream https://api.openai.com/v1             # real OpenAI
uv run prompt-cache-meter proxy --generation legacy --price-as gpt-5-mini        # older rules, cheaper model
```

## 2. Point your app at the proxy

The base URL is always:

```
http://127.0.0.1:8787/v1
```

Keep the `/v1`. Without it, calls are still forwarded but **not simulated** (see [Troubleshooting](#7-troubleshooting)).

The API key is passed through to the upstream unchanged:

- **Ollama:** any non-empty string works.
- **OpenRouter:** your OpenRouter key, with OpenRouter model ids.
- **OpenAI:** your OpenAI key.

### Easiest: an environment variable

The official OpenAI SDKs (Python and Node) read `OPENAI_BASE_URL`, and so do many tools built on them. If your app does not set a base URL in code, this is often all you need:

```sh
export OPENAI_BASE_URL=http://127.0.0.1:8787/v1
export OPENAI_API_KEY=ollama        # or your real key, depending on the upstream
python your_app.py
```

### Python, OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8787/v1", api_key="ollama")
reply = client.chat.completions.create(
    model="qwen2.5:14b",
    messages=[{"role": "system", "content": "..."}, {"role": "user", "content": "..."}],
)
```

Streaming (`stream=True`) works too.

### Node / TypeScript, OpenAI SDK

```ts
import OpenAI from "openai";

const client = new OpenAI({ baseURL: "http://127.0.0.1:8787/v1", apiKey: "ollama" });
const reply = await client.chat.completions.create({ model: "qwen2.5:14b", messages });
```

### LangChain (Python)

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(base_url="http://127.0.0.1:8787/v1", api_key="ollama", model="qwen2.5:14b")
```

### LlamaIndex

Use `OpenAILike`, which accepts model names that are not OpenAI's:

```python
from llama_index.llms.openai_like import OpenAILike

llm = OpenAILike(api_base="http://127.0.0.1:8787/v1", api_key="ollama", model="qwen2.5:14b", is_chat_model=True)
```

### LiteLLM

```python
import litellm

litellm.completion(model="openai/qwen2.5:14b", api_base="http://127.0.0.1:8787/v1", api_key="ollama", messages=messages)
```

### Frameworks that default to the Responses API

The proxy only simulates **Chat Completions** (`/v1/chat/completions`). Some newer frameworks use the Responses API by default. Switch them to Chat Completions:

- **OpenAI Agents SDK:** call `set_default_openai_api("chat_completions")`, or use `OpenAIChatCompletionsModel`.
- **Vercel AI SDK:** use `openai.chat("model")` instead of `openai("model")`.

### curl

```sh
curl http://127.0.0.1:8787/v1/chat/completions \
  -H "Content-Type: application/json" -H "Authorization: Bearer ollama" \
  -d '{"model": "qwen2.5:14b", "messages": [{"role": "user", "content": "hi"}]}'
```

### Anything else

If a tool lets you set an "OpenAI-compatible base URL", "API base" or "custom endpoint", set it to `http://127.0.0.1:8787/v1`.

## 3. Read the console line

Each successful chat call prints one line:

```
06:40:02 qwen2.5:14b as gpt-6.1-terra key=- | in 1,353 cached 1,350 written 0 1x 3 hit 100% $0.000210 | upstream in 1,535 cached 1,534 out 30 | 2.48s
```

| Part | Meaning |
| --- | --- |
| `06:40:02` | Time of the call (UTC). |
| `qwen2.5:14b as gpt-6.1-terra` | The model your app asked for, and the OpenAI model whose prices were used. |
| `key=-` | The `prompt_cache_key` the app sent (`-` = none). Each key has its own simulated cache. |
| `in 1,353` | Input tokens, counted the way OpenAI counts them (tiktoken). |
| `cached 1,350` | Tokens OpenAI would read from cache (cheap: 0.05x to 0.1x the input price). |
| `written 0` | Tokens OpenAI would write to cache (new rules: 1.25x the input price). |
| `1x 3` | Tokens billed at the normal input price. |
| `hit 100%` | Cached as a share of input. |
| `$0.000210` | Simulated cost of the call, output included. |
| `upstream in ... cached ... out ...` | What the real provider reported. Its token counts differ because it uses its own tokenizer. `upstream usage -` means it sent none (common for streamed Ollama calls). |
| `2.48s` | Real time for the call. |
| yellow text | Warnings, such as a request the simulator could not read. The call itself still went through. |

What to look for:

- **First call of a prompt:** `written` is high and `cached` is 0. That is a cold cache.
- **Next calls:** `cached` should cover the shared start of the prompt. If it stays 0, something at the start changes on every call, such as a timestamp, a user name or a reordered tool list.
- **Low hit rate in a chat:** check whether the app trims or rewrites old messages. Dropping the oldest turn changes the prefix every time.

## 4. Analyse the log

Every call is also appended to `runs/proxy/<YYYYMMDD>.jsonl`, one JSON object per line. It holds the models, `prompt_cache_key`, the full simulated bill, the upstream's usage, latency, whether the call streamed, and any warnings. **No message content or headers are logged.**

Totals for a day with `jq`:

```sh
jq -s 'map(select(.bill)) | {
  calls: length,
  input: (map(.bill.input_tokens) | add),
  cached: (map(.bill.cached_tokens) | add),
  cost_usd: (map(.bill.cost_usd) | add)
}' runs/proxy/20261001.jsonl
```

Per `prompt_cache_key`:

```sh
jq -s 'map(select(.bill)) | group_by(.prompt_cache_key) | map({
  key: .[0].prompt_cache_key, calls: length, cost_usd: (map(.bill.cost_usd) | add)
})' runs/proxy/20261001.jsonl
```

To compare designs, run the app once per design against a freshly started proxy, then compare the totals.

## 5. What is and is not simulated

| Simulated | Not simulated (forwarded unchanged) |
| --- | --- |
| `POST /v1/chat/completions`, plain and streamed | The Responses API, embeddings, models list and every other endpoint |
| Roles: system, developer, user, assistant, tool | Explicit breakpoints and the 12h TTL (OpenAI offers these only on the Responses API) |
| Tools, `response_format`, `reasoning_effort`, `verbosity`, `parallel_tool_calls` (changing them breaks the cache) | Failed calls (non-2xx): passed back to the app, not billed, cache untouched |
| `prompt_cache_key` (separate cache per key) | |
| `prompt_cache_retention` (legacy rules: `in_memory` or `24h`) | |

Things to keep in mind:

- **The cache starts empty** each time the proxy starts, and lives only in memory.
- **TTLs run on real time.** With the new rules, a prompt not reused for 30 minutes is a miss again.
- **Images and audio** are counted approximately and flagged with a warning.
- **Overlapping calls** are simulated in the order they finish, which can differ slightly from the order they arrived.
- The rules and prices behind the numbers are in [`src/prompt_cache_meter/cache_rules.toml`](src/prompt_cache_meter/cache_rules.toml). Correct them there, or pass your own file with `--rules`.

## 6. Docker and other machines

**App in Docker, proxy on your Mac:** use `http://host.docker.internal:8787/v1` as the base URL, and start the proxy with `--host 0.0.0.0`.

**App on another machine:** start the proxy with `--host 0.0.0.0` and use `http://<your-ip>:8787/v1`.

The proxy has no authentication of its own. When it listens on `0.0.0.0`, anyone on your network can use it to reach your upstream, with whatever key they send. Only do this on a network you trust, and stop the proxy when you are done.

## 7. Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| The app works, but no lines appear | The base URL is missing `/v1`, or the app uses the Responses API | Use `http://127.0.0.1:8787/v1`; switch the app to Chat Completions (section 2) |
| The app gets a `502 bad_gateway` | The upstream is not reachable | Check the provider is running: `curl http://127.0.0.1:11434/v1/models` for Ollama |
| The app gets 401 / 403 | Wrong key for that upstream | The key goes straight through; use the upstream's key |
| `not simulated: TranslationError` | The request has something the simulator cannot read, such as an unknown role | The call still worked; only the simulation was skipped |
| `prompt_cache_retention ... not supported` | The app sent a retention value the selected rules do not have | Harmless; the default TTL was used. Try `--generation legacy` if the app targets older models |
| `cached` is always 0 | The prompt is under 1,024 tokens, or its start changes on every call | Check prompt length; look for timestamps or per-user text near the top |
| `address already in use` | Another proxy is running on that port | Stop it, or use `--port 8788` |
| Cost looks wrong for your model | The model is priced as `--price-as` | Add your model to `[prices]` in the rules file, or pass `--price-as` |
| Ollama answers look cut off or confused on long prompts | Ollama silently truncates past its context window | Start Ollama with `OLLAMA_CONTEXT_LENGTH=16384 ollama serve` |
