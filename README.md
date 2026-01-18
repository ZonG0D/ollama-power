# ollama-power

Single-file, zero-dependency Python powerhouse for Ollama: full model management, embeddings, lightweight SQLite RAG, agent/tool loop, interactive REPL, structured outputs, and streaming. All packaged in one Python script.

Default Ollama server: `http://localhost:11434`

---

## Features

- **Single file, stdlib-only** (no pip dependencies)
- **Model management**
  - list models (`tags`)
  - list running models (`ps`)
  - show model details (`show`)
  - pull/push (streaming progress)
  - copy, create, delete
- **Chat**
  - `/api/chat` streaming NDJSON
  - optional **thinking** output (model-dependent)
  - optional **tool calling** loop (agent-style)
  - optional **structured outputs** (`format: "json"` or JSON schema)
- **Embeddings + SQLite RAG**
  - embed text
  - ingest files into SQLite (`chunks` table)
  - cosine-similarity retrieval
  - context-only answering mode with chunk_id citations
- **Interactive REPL**
  - toggle tools/stream/thinking/format at runtime
  - optional image attach (base64)

---

## Requirements

- Python 3
- A running Ollama server at `localhost:11434`
- At least:
  - one chat model (example: `qwen3`, `gemma3`)
  - one embedding model (example: `embeddinggemma`)

---

## Install

Place `ollama_power.py` in a directory, then run:

```bash
chmod +x ollama_power.py
./ollama_power.py tags
```

---

## Configuration

### Change the Ollama host

Edit the constant near the top of the script:

```python
BASE_URL = "http://localhost:11434"
```

---

## Command reference

### List local models

```bash
python3 ollama_power.py tags
```

### List running models

```bash
python3 ollama_power.py ps
```

### Show model details

```bash
python3 ollama_power.py show gemma3
```

### Pull a model (stream progress)

```bash
python3 ollama_power.py pull gemma3
python3 ollama_power.py pull gemma3 --insecure
```

### Push a model (stream progress)

```bash
python3 ollama_power.py push username/model
python3 ollama_power.py push username/model --insecure
```

### Copy a model

```bash
python3 ollama_power.py copy gemma3 gemma3-backup
```

### Create a model (stream progress)

```bash
python3 ollama_power.py create mario \
  --from-model gemma3 \
  --system "Roleplay as Mario."
```

Optional flags supported by the script:

- `--from-model`
- `--system`
- `--template`
- `--license`
- `--parameters-json` (JSON string)

### Delete a model

```bash
python3 ollama_power.py delete gemma3
```

---

## Embeddings

### Embed a single string

```bash
python3 ollama_power.py embed --model embeddinggemma --text "hello"
```

### Embed multiple lines (batch mode)

`--batch` splits `--text` by newline and sends a list.

```bash
python3 ollama_power.py embed --model embeddinggemma --text $'line1\nline2\nline3' --batch
```

---

## SQLite RAG

### Ingest documents

Creates or updates the SQLite DB and stores chunk embeddings.

```bash
python3 ollama_power.py ingest \
  --db rag.db \
  --embed-model embeddinggemma \
  --max-chars 1200 \
  --batch-size 32 \
  docs/*.md
```

**DB schema (created automatically):**

- `chunks`
  - `chunk_id` (autoincrement)
  - `source` (file path)
  - `chunk` (text)
  - `embedding` (JSON-encoded float list)

Chunking behavior:

- split on blank lines
- pack into `max-chars` windows
- hard-split oversize segments into fixed slices

### Ask questions over the DB

Retrieves top-k chunks, builds a context prompt, then runs `/api/chat`.

```bash
python3 ollama_power.py rag \
  --db rag.db \
  --embed-model embeddinggemma \
  --chat-model qwen3 \
  --query "create a table of contents" \
  --top-k 6
```

---

## Agent/tool loop

Enable with `--tools`. Tool calls are executed locally, then appended back into the chat as `role:"tool"` messages.

```bash
python3 ollama_power.py rag \
  --db rag.db \
  --embed-model embeddinggemma \
  --chat-model qwen3 \
  --query "summarize and compute token budget" \
  --tools \
  --stream \
  --think true \
  --max-steps 8
```

### Built-in tools

- `time_now` — UTC timestamp (ISO 8601)
- `calc` — safe arithmetic evaluator (no names, no calls)
- `fs_read_text` — read a local text file (char limit)
- `fs_list_dir` — list directory entries (limit)
- `rag_search` — query the same SQLite store

**Security warning:** tools allow local file reads and directory listing. Keep `--tools` off for untrusted prompts.

---

## Streaming

Add `--stream` on `rag` or `repl` to print token output as NDJSON chunks arrive.

```bash
python3 ollama_power.py rag --db rag.db --embed-model embeddinggemma --chat-model qwen3 --query "..." --stream
```

Optional: `--print-thinking` prints thinking and answer sections separately during streaming.

---

## Thinking

`--think` is passed through to `/api/chat`. Support depends on the chosen model.

Examples:

```bash
python3 ollama_power.py repl --model qwen3 --think true --stream
python3 ollama_power.py repl --model gpt-oss --think high --stream
```

---

## Structured outputs

### Force JSON

```bash
python3 ollama_power.py rag \
  --db rag.db \
  --embed-model embeddinggemma \
  --chat-model qwen3 \
  --query "extract action items" \
  --format-json
```

### Enforce a JSON schema

Create `schema.json`:

```json
{
  "type": "object",
  "properties": {
    "title": { "type": "string" },
    "items": {
      "type": "array",
      "items": { "type": "string" }
    }
  },
  "required": ["title", "items"],
  "additionalProperties": false
}
```

Run:

```bash
python3 ollama_power.py rag \
  --db rag.db \
  --embed-model embeddinggemma \
  --chat-model qwen3 \
  --query "produce a checklist" \
  --schema schema.json
```

---

## Options passthrough

`--options-json` forwards an `options` object directly to `/api/chat`.

```bash
python3 ollama_power.py repl \
  --model qwen3 \
  --options-json '{"num_ctx":8192,"num_predict":512}'
```

---

## REPL

Start:

```bash
python3 ollama_power.py repl --model qwen3 --stream
```

Slash commands:

- `/quit`
- `/reset`
- `/tools on|off`
- `/stream on|off`
- `/think true|false|<string>`
- `/format json|off`

Attach an image to the next user message:

```bash
python3 ollama_power.py repl --model qwen3 --image ./screenshot.png
```

---

## Metrics

After each chat run, compact timing and token metrics print to stderr when present (durations converted to milliseconds).

---

## Troubleshooting

### Connection refused or timeouts

- Confirm Ollama is running on `localhost:11434`
- Confirm Ollama is bound to a reachable interface if using a remote host

### Streaming stops mid-response

- Inspect stderr for an `"error"` field surfaced from a stream chunk
- Retry without tools to isolate tool-loop effects

### RAG feels slow at scale

- Current retrieval loads all embeddings and cosine-scores in Python
- For large corpora, replace the SQLite brute-force scan with an indexed vector backend

---

## Project layout

Minimal layout:

```text
ollama-power/
  ollama_power.py
  README.md
```

Optional:

```text
ollama-power/
  ollama_power.py
  README.md
  schema.json
  rag.db
  docs/
```
