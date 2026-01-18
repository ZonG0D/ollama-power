#!/usr/bin/env python3
# ollama_power.py
# Single-file Ollama client for http://localhost:11434

from __future__ import annotations

import argparse
import ast
import base64
import datetime as dt
import json
import math
import os
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


BASE_URL = "http://localhost:11434"


class OllamaHTTPError(RuntimeError):
    pass


def _json_dumps(obj: Any) -> bytes:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _http_request(
    method: str,
    path: str,
    payload: Optional[Dict[str, Any]] = None,
    timeout_s: float = 300.0,
) -> urllib.response.addinfourl:
    url = f"{BASE_URL}{path}"
    data = None if payload is None else _json_dumps(payload)
    req = urllib.request.Request(url=url, data=data, method=method.upper())
    req.add_header("Content-Type", "application/json")
    try:
        return urllib.request.urlopen(req, timeout=timeout_s)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(body)
            msg = parsed.get("error") or body
        except Exception:
            msg = body
        raise OllamaHTTPError(f"HTTP {exc.code} {path}: {msg}") from exc
    except urllib.error.URLError as exc:
        raise OllamaHTTPError(f"Network error {path}: {exc}") from exc


def http_json(
    method: str,
    path: str,
    payload: Optional[Dict[str, Any]] = None,
    timeout_s: float = 300.0,
) -> Dict[str, Any]:
    resp = _http_request(method, path, payload, timeout_s)
    raw = resp.read().decode("utf-8", errors="replace")
    try:
        parsed = json.loads(raw)
    except Exception as exc:
        raise OllamaHTTPError(f"Invalid JSON from {path}: {raw[:4000]}") from exc
    if isinstance(parsed, dict) and "error" in parsed:
        raise OllamaHTTPError(f"{path}: {parsed['error']}")
    return parsed


def http_ndjson_stream(
    method: str,
    path: str,
    payload: Dict[str, Any],
    timeout_s: float = 300.0,
) -> Iterable[Dict[str, Any]]:
    resp = _http_request(method, path, payload, timeout_s)
    while True:
        line = resp.readline()
        if not line:
            break
        stripped = line.strip()
        if not stripped:
            continue
        try:
            parsed = json.loads(stripped.decode("utf-8", errors="replace"))
        except Exception:
            raise OllamaHTTPError(f"Invalid NDJSON line from {path}: {stripped[:4000]}")
        if isinstance(parsed, dict) and "error" in parsed:
            raise OllamaHTTPError(f"{path}: {parsed['error']}")
        yield parsed


def b64_file(path: str) -> str:
    with open(path, "rb") as handle:
        return base64.b64encode(handle.read()).decode("ascii")


def _ns_to_ms(ns_val: Optional[int]) -> Optional[float]:
    if ns_val is None:
        return None
    return float(ns_val) / 1_000_000.0


def _print_usage_metrics(obj: Dict[str, Any]) -> None:
    fields = [
        ("total_duration", "total_ms"),
        ("load_duration", "load_ms"),
        ("prompt_eval_duration", "prompt_eval_ms"),
        ("eval_duration", "eval_ms"),
        ("prompt_eval_count", "prompt_tokens"),
        ("eval_count", "output_tokens"),
        ("done_reason", "done_reason"),
    ]
    out: Dict[str, Any] = {}
    for key, label in fields:
        if key not in obj:
            continue
        if key.endswith("_duration"):
            out[label] = _ns_to_ms(obj.get(key))
        else:
            out[label] = obj.get(key)
    if out:
        sys.stderr.write("\n")
        sys.stderr.write(json.dumps(out, indent=2))
        sys.stderr.write("\n")


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: Dict[str, Any]


def _tool_def(spec: ToolSpec) -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.parameters,
        },
    }


def _safe_calc(expr: str) -> str:
    node = ast.parse(expr, mode="eval")

    allowed = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
        ast.USub,
        ast.UAdd,
        ast.Constant,
        ast.Call,
        ast.Name,
    )

    for subnode in ast.walk(node):
        if not isinstance(subnode, allowed):
            raise ValueError("Unsupported expression")
        if isinstance(subnode, ast.Call):
            raise ValueError("Calls not allowed")
        if isinstance(subnode, ast.Name):
            raise ValueError("Names not allowed")

    val = eval(compile(node, "<calc>", "eval"), {"__builtins__": {}})
    return str(val)


def tool_time_now(_: Dict[str, Any]) -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def tool_calc(args: Dict[str, Any]) -> str:
    expr = str(args.get("expr", "")).strip()
    if not expr:
        return "Error: empty expression"
    try:
        return _safe_calc(expr)
    except Exception as exc:
        return f"Error: {exc}"


def tool_fs_read_text(args: Dict[str, Any]) -> str:
    path = str(args.get("path", "")).strip()
    limit = int(args.get("limit_chars", 20000))
    if not path:
        return "Error: empty path"
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            data = handle.read(limit)
        return data
    except Exception as exc:
        return f"Error: {exc}"


def tool_fs_list_dir(args: Dict[str, Any]) -> str:
    path = str(args.get("path", ".")).strip() or "."
    limit = int(args.get("limit", 200))
    try:
        entries = sorted(os.listdir(path))[:limit]
        return json.dumps({"path": path, "entries": entries}, ensure_ascii=False, indent=2)
    except Exception as exc:
        return f"Error: {exc}"


def dot(vec_a: List[float], vec_b: List[float]) -> float:
    total = 0.0
    max_len = min(len(vec_a), len(vec_b))
    for pos in range(max_len):
        total += float(vec_a[pos]) * float(vec_b[pos])
    return total


def norm(vec: List[float]) -> float:
    return math.sqrt(dot(vec, vec))


def cosine(vec_a: List[float], vec_b: List[float]) -> float:
    denom = norm(vec_a) * norm(vec_b)
    if denom == 0.0:
        return 0.0
    return dot(vec_a, vec_b) / denom


def ensure_db(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
              chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
              source TEXT NOT NULL,
              chunk TEXT NOT NULL,
              embedding TEXT NOT NULL
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source)")
        conn.commit()
    finally:
        conn.close()


def embed(
    embed_model: str,
    text_or_list: Any,
    keep_alive: Optional[str] = None,
    truncate: Optional[bool] = None,
    dimensions: Optional[int] = None,
    options: Optional[Dict[str, Any]] = None,
) -> List[List[float]]:
    payload: Dict[str, Any] = {"model": embed_model, "input": text_or_list}
    if keep_alive is not None:
        payload["keep_alive"] = keep_alive
    if truncate is not None:
        payload["truncate"] = bool(truncate)
    if dimensions is not None:
        payload["dimensions"] = int(dimensions)
    if options is not None:
        payload["options"] = options
    resp = http_json("POST", "/api/embed", payload)
    embs = resp.get("embeddings")
    if not isinstance(embs, list) or not embs:
        raise OllamaHTTPError("Missing embeddings in /api/embed response")
    out: List[List[float]] = []
    for row in embs:
        out.append([float(val) for val in row])
    return out


def chunk_text(text: str, max_chars: int) -> List[str]:
    cleaned = text.replace("\r\n", "\n")
    parts = [seg.strip() for seg in cleaned.split("\n\n") if seg.strip()]
    out: List[str] = []
    buf = ""
    for seg in parts:
        if not buf:
            buf = seg
            continue
        if len(buf) + 2 + len(seg) <= max_chars:
            buf = buf + "\n\n" + seg
        else:
            out.append(buf)
            buf = seg
    if buf:
        out.append(buf)

    final_out: List[str] = []
    for seg in out:
        if len(seg) <= max_chars:
            final_out.append(seg)
            continue
        start = 0
        while start < len(seg):
            end = min(start + max_chars, len(seg))
            final_out.append(seg[start:end])
            start = end
    return final_out


def ingest_files(
    db_path: str,
    embed_model: str,
    file_paths: List[str],
    max_chars: int = 1200,
    batch_size: int = 32,
) -> None:
    ensure_db(db_path)
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        for path in file_paths:
            with open(path, "r", encoding="utf-8", errors="replace") as handle:
                data = handle.read()
            chunks = chunk_text(data, max_chars=max_chars)

            idx = 0
            while idx < len(chunks):
                batch = chunks[idx : idx + batch_size]
                idx += batch_size
                embs = embed(embed_model, batch)
                for subpos, chunk_val in enumerate(batch):
                    emb_val = embs[subpos]
                    cur.execute(
                        "INSERT INTO chunks(source, chunk, embedding) VALUES(?,?,?)",
                        (path, chunk_val, json.dumps(emb_val, separators=(",", ":"))),
                    )
                conn.commit()
                sys.stderr.write(f"ingest: {path} +{len(batch)} chunks\n")
    finally:
        conn.close()


def search_db(
    db_path: str,
    embed_model: str,
    query_text: str,
    top_k: int = 6,
) -> List[Dict[str, Any]]:
    ensure_db(db_path)
    q_emb = embed(embed_model, query_text)[0]
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute("SELECT chunk_id, source, chunk, embedding FROM chunks")
        rows = cur.fetchall()
    finally:
        conn.close()

    scored: List[Tuple[float, int, str, str]] = []
    for row in rows:
        chunk_id = int(row[0])
        source = str(row[1])
        chunk_val = str(row[2])
        emb_val = json.loads(row[3])
        score = cosine(q_emb, [float(v) for v in emb_val])
        scored.append((score, chunk_id, source, chunk_val))

    scored.sort(key=lambda tup: tup[0], reverse=True)
    out: List[Dict[str, Any]] = []
    for score, chunk_id, source, chunk_val in scored[: max(1, top_k)]:
        out.append(
            {
                "chunk_id": chunk_id,
                "source": source,
                "score": float(score),
                "chunk": chunk_val,
            }
        )
    return out


def tool_rag_search(args: Dict[str, Any]) -> str:
    db_path = str(args.get("db_path", "rag.db")).strip() or "rag.db"
    embed_model = str(args.get("embed_model", "embeddinggemma")).strip() or "embeddinggemma"
    query_text = str(args.get("query", "")).strip()
    top_k = int(args.get("top_k", 6))
    if not query_text:
        return "Error: empty query"
    try:
        hits = search_db(db_path, embed_model, query_text, top_k=top_k)
        return json.dumps({"hits": hits}, ensure_ascii=False, indent=2)
    except Exception as exc:
        return f"Error: {exc}"


TOOL_SPECS: List[ToolSpec] = [
    ToolSpec(
        name="time_now",
        description="Return current UTC timestamp in ISO 8601.",
        parameters={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    ),
    ToolSpec(
        name="calc",
        description="Evaluate a basic arithmetic expression.",
        parameters={
            "type": "object",
            "properties": {"expr": {"type": "string"}},
            "required": ["expr"],
            "additionalProperties": False,
        },
    ),
    ToolSpec(
        name="fs_read_text",
        description="Read a local text file.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "limit_chars": {"type": "integer", "minimum": 1, "maximum": 200000},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    ),
    ToolSpec(
        name="fs_list_dir",
        description="List directory entries.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 2000},
            },
            "additionalProperties": False,
        },
    ),
    ToolSpec(
        name="rag_search",
        description="Search a local SQLite embedding store.",
        parameters={
            "type": "object",
            "properties": {
                "db_path": {"type": "string"},
                "embed_model": {"type": "string"},
                "query": {"type": "string"},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 50},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    ),
]


TOOL_IMPL = {
    "time_now": tool_time_now,
    "calc": tool_calc,
    "fs_read_text": tool_fs_read_text,
    "fs_list_dir": tool_fs_list_dir,
    "rag_search": tool_rag_search,
}


def chat_once_stream(
    model: str,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    think: Optional[Any] = None,
    keep_alive: Optional[Any] = None,
    fmt: Optional[Any] = None,
    options: Optional[Dict[str, Any]] = None,
    logprobs: Optional[bool] = None,
    top_logprobs: Optional[int] = None,
    print_thinking: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    payload: Dict[str, Any] = {"model": model, "messages": messages, "stream": True}
    if tools is not None:
        payload["tools"] = tools
    if think is not None:
        payload["think"] = think
    if keep_alive is not None:
        payload["keep_alive"] = keep_alive
    if fmt is not None:
        payload["format"] = fmt
    if options is not None:
        payload["options"] = options
    if logprobs is not None:
        payload["logprobs"] = bool(logprobs)
    if top_logprobs is not None:
        payload["top_logprobs"] = int(top_logprobs)

    thinking_buf = ""
    content_buf = ""
    tool_calls: List[Dict[str, Any]] = []
    final_chunk: Dict[str, Any] = {}

    seen_thinking = False
    seen_content = False

    for chunk in http_ndjson_stream("POST", "/api/chat", payload):
        final_chunk = chunk
        msg = chunk.get("message") or {}

        think_part = msg.get("thinking")
        if isinstance(think_part, str) and think_part:
            thinking_buf += think_part
            if print_thinking:
                if not seen_thinking:
                    sys.stdout.write("\n[thinking]\n")
                    seen_thinking = True
                sys.stdout.write(think_part)
                sys.stdout.flush()

        cont_part = msg.get("content")
        if isinstance(cont_part, str) and cont_part:
            content_buf += cont_part
            if not seen_content:
                if print_thinking and seen_thinking:
                    sys.stdout.write("\n\n[answer]\n")
                seen_content = True
            sys.stdout.write(cont_part)
            sys.stdout.flush()

        tc_part = msg.get("tool_calls")
        if isinstance(tc_part, list) and tc_part:
            for call_obj in tc_part:
                tool_calls.append(call_obj)

        if chunk.get("done") is True:
            break

    sys.stdout.write("\n")
    sys.stdout.flush()

    assistant_msg: Dict[str, Any] = {"role": "assistant"}
    if thinking_buf:
        assistant_msg["thinking"] = thinking_buf
    if content_buf:
        assistant_msg["content"] = content_buf
    if tool_calls:
        assistant_msg["tool_calls"] = tool_calls

    return assistant_msg, final_chunk


def chat_once_json(
    model: str,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    think: Optional[Any] = None,
    keep_alive: Optional[Any] = None,
    fmt: Optional[Any] = None,
    options: Optional[Dict[str, Any]] = None,
    logprobs: Optional[bool] = None,
    top_logprobs: Optional[int] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"model": model, "messages": messages, "stream": False}
    if tools is not None:
        payload["tools"] = tools
    if think is not None:
        payload["think"] = think
    if keep_alive is not None:
        payload["keep_alive"] = keep_alive
    if fmt is not None:
        payload["format"] = fmt
    if options is not None:
        payload["options"] = options
    if logprobs is not None:
        payload["logprobs"] = bool(logprobs)
    if top_logprobs is not None:
        payload["top_logprobs"] = int(top_logprobs)
    return http_json("POST", "/api/chat", payload)


def run_agent_loop(
    model: str,
    messages: List[Dict[str, Any]],
    enable_tools: bool,
    stream: bool,
    think: Optional[Any],
    fmt: Optional[Any],
    options: Optional[Dict[str, Any]],
    keep_alive: Optional[Any],
    max_steps: int,
    print_thinking: bool,
) -> None:
    tool_defs = [_tool_def(spec) for spec in TOOL_SPECS] if enable_tools else None

    step = 0
    while True:
        step += 1
        if step > max_steps:
            raise OllamaHTTPError("Agent loop exceeded max_steps")

        if stream:
            assistant_msg, final_chunk = chat_once_stream(
                model=model,
                messages=messages,
                tools=tool_defs,
                think=think,
                keep_alive=keep_alive,
                fmt=fmt,
                options=options,
                print_thinking=print_thinking,
            )
            messages.append(assistant_msg)
            _print_usage_metrics(final_chunk)
            tool_calls = assistant_msg.get("tool_calls") or []
        else:
            resp = chat_once_json(
                model=model,
                messages=messages,
                tools=tool_defs,
                think=think,
                keep_alive=keep_alive,
                fmt=fmt,
                options=options,
            )
            msg = resp.get("message") or {}
            messages.append(msg)
            text = msg.get("content") or ""
            if isinstance(text, str) and text:
                sys.stdout.write(text + "\n")
            _print_usage_metrics(resp)
            tool_calls = msg.get("tool_calls") or []

        if not tool_calls:
            break

        for call_obj in tool_calls:
            fn_obj = call_obj.get("function") or {}
            fn_name = fn_obj.get("name")
            fn_args = fn_obj.get("arguments") or {}
            if not isinstance(fn_name, str) or fn_name not in TOOL_IMPL:
                result = "Error: unknown tool"
                tool_name = str(fn_name) if fn_name else "unknown"
            else:
                tool_name = fn_name
                try:
                    result = TOOL_IMPL[fn_name](fn_args if isinstance(fn_args, dict) else {})
                except Exception as exc:
                    result = f"Error: {exc}"
            messages.append({"role": "tool", "tool_name": tool_name, "content": str(result)})


def cmd_tags(_: argparse.Namespace) -> None:
    resp = http_json("GET", "/api/tags", None)
    sys.stdout.write(json.dumps(resp, indent=2))
    sys.stdout.write("\n")


def cmd_ps(_: argparse.Namespace) -> None:
    resp = http_json("GET", "/api/ps", None)
    sys.stdout.write(json.dumps(resp, indent=2))
    sys.stdout.write("\n")


def cmd_show(args: argparse.Namespace) -> None:
    resp = http_json("POST", "/api/show", {"model": args.model})
    sys.stdout.write(json.dumps(resp, indent=2))
    sys.stdout.write("\n")


def cmd_delete(args: argparse.Namespace) -> None:
    resp = http_json("DELETE", "/api/delete", {"model": args.model})
    sys.stdout.write(json.dumps(resp, indent=2))
    sys.stdout.write("\n")


def _status_stream(path: str, payload: Dict[str, Any]) -> None:
    for chunk in http_ndjson_stream("POST", path, payload):
        sys.stdout.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        if chunk.get("status") == "success":
            break


def cmd_pull(args: argparse.Namespace) -> None:
    payload: Dict[str, Any] = {"model": args.model, "stream": True}
    if args.insecure:
        payload["insecure"] = True
    _status_stream("/api/pull", payload)


def cmd_push(args: argparse.Namespace) -> None:
    payload: Dict[str, Any] = {"model": args.model, "stream": True}
    if args.insecure:
        payload["insecure"] = True
    _status_stream("/api/push", payload)


def cmd_copy(args: argparse.Namespace) -> None:
    resp = http_json("POST", "/api/copy", {"source": args.source, "destination": args.destination})
    sys.stdout.write(json.dumps(resp, indent=2))
    sys.stdout.write("\n")


def cmd_create(args: argparse.Namespace) -> None:
    payload: Dict[str, Any] = {"model": args.model, "stream": True}
    if args.from_model is not None:
        payload["from"] = args.from_model
    if args.system is not None:
        payload["system"] = args.system
    if args.template is not None:
        payload["template"] = args.template
    if args.license is not None:
        payload["license"] = args.license
    if args.parameters_json is not None:
        payload["parameters"] = json.loads(args.parameters_json)
    _status_stream("/api/create", payload)


def cmd_embed(args: argparse.Namespace) -> None:
    inp: Any
    if args.batch:
        inp = [line for line in args.text.split("\\n") if line.strip()]
    else:
        inp = args.text
    embs = embed(args.model, inp)
    sys.stdout.write(json.dumps({"embeddings": embs}, ensure_ascii=False))
    sys.stdout.write("\n")


def _load_schema(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def cmd_ingest(args: argparse.Namespace) -> None:
    ingest_files(
        db_path=args.db,
        embed_model=args.embed_model,
        file_paths=args.files,
        max_chars=args.max_chars,
        batch_size=args.batch_size,
    )


def cmd_rag(args: argparse.Namespace) -> None:
    hits = search_db(args.db, args.embed_model, args.query, top_k=args.top_k)
    context_lines: List[str] = []
    for hit in hits:
        context_lines.append(f"[chunk_id={hit['chunk_id']} source={hit['source']} score={hit['score']:.4f}]")
        context_lines.append(hit["chunk"])
        context_lines.append("")
    context = "\n".join(context_lines).strip()

    system_msg = (
        "Answer using only the provided context. "
        "If context is insufficient, say so. "
        "Include chunk_id citations when relevant."
    )

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": f"Context:\n{context}\n\nQuery:\n{args.query}"},
    ]

    fmt: Optional[Any] = None
    if args.format_json:
        fmt = "json"
    if args.schema is not None:
        fmt = _load_schema(args.schema)

    opts = json.loads(args.options_json) if args.options_json else None
    run_agent_loop(
        model=args.chat_model,
        messages=messages,
        enable_tools=args.tools,
        stream=args.stream,
        think=args.think,
        fmt=fmt,
        options=opts,
        keep_alive=args.keep_alive,
        max_steps=args.max_steps,
        print_thinking=args.print_thinking,
    )


def cmd_repl(args: argparse.Namespace) -> None:
    fmt: Optional[Any] = None
    if args.format_json:
        fmt = "json"
    if args.schema is not None:
        fmt = _load_schema(args.schema)

    opts = json.loads(args.options_json) if args.options_json else None

    messages: List[Dict[str, Any]] = []
    if args.system is not None:
        messages.append({"role": "system", "content": args.system})

    sys.stderr.write("REPL: /quit /reset /tools on|off /think val /stream on|off /format json|off\n")
    enable_tools = args.tools
    stream = args.stream
    think: Any = args.think
    cur_fmt: Optional[Any] = fmt

    while True:
        line = sys.stdin.readline()
        if not line:
            break
        text = line.rstrip("\n")
        if not text:
            continue

        if text.startswith("/"):
            parts = text.split()
            cmd = parts[0].lower()
            if cmd == "/quit":
                break
            if cmd == "/reset":
                messages = [{"role": "system", "content": args.system}] if args.system else []
                sys.stderr.write("state reset\n")
                continue
            if cmd == "/tools" and len(parts) >= 2:
                enable_tools = parts[1].lower() == "on"
                sys.stderr.write(f"tools={enable_tools}\n")
                continue
            if cmd == "/stream" and len(parts) >= 2:
                stream = parts[1].lower() == "on"
                sys.stderr.write(f"stream={stream}\n")
                continue
            if cmd == "/think" and len(parts) >= 2:
                val = parts[1].lower()
                if val in ("true", "false"):
                    think = (val == "true")
                else:
                    think = parts[1]
                sys.stderr.write(f"think={think}\n")
                continue
            if cmd == "/format" and len(parts) >= 2:
                val = parts[1].lower()
                if val == "json":
                    cur_fmt = "json"
                elif val == "off":
                    cur_fmt = None
                sys.stderr.write(f"format={cur_fmt}\n")
                continue
            sys.stderr.write("unknown command\n")
            continue

        user_msg: Dict[str, Any] = {"role": "user", "content": text}
        if args.image is not None:
            user_msg["images"] = [b64_file(args.image)]
        messages.append(user_msg)

        run_agent_loop(
            model=args.model,
            messages=messages,
            enable_tools=enable_tools,
            stream=stream,
            think=think,
            fmt=cur_fmt,
            options=opts,
            keep_alive=args.keep_alive,
            max_steps=args.max_steps,
            print_thinking=args.print_thinking,
        )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ollama_power.py")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("tags")
    sp.set_defaults(func=cmd_tags)

    sp = sub.add_parser("ps")
    sp.set_defaults(func=cmd_ps)

    sp = sub.add_parser("show")
    sp.add_argument("model")
    sp.set_defaults(func=cmd_show)

    sp = sub.add_parser("delete")
    sp.add_argument("model")
    sp.set_defaults(func=cmd_delete)

    sp = sub.add_parser("pull")
    sp.add_argument("model")
    sp.add_argument("--insecure", action="store_true")
    sp.set_defaults(func=cmd_pull)

    sp = sub.add_parser("push")
    sp.add_argument("model")
    sp.add_argument("--insecure", action="store_true")
    sp.set_defaults(func=cmd_push)

    sp = sub.add_parser("copy")
    sp.add_argument("source")
    sp.add_argument("destination")
    sp.set_defaults(func=cmd_copy)

    sp = sub.add_parser("create")
    sp.add_argument("model")
    sp.add_argument("--from-model", dest="from_model")
    sp.add_argument("--system")
    sp.add_argument("--template")
    sp.add_argument("--license")
    sp.add_argument("--parameters-json")
    sp.set_defaults(func=cmd_create)

    sp = sub.add_parser("embed")
    sp.add_argument("--model", required=True)
    sp.add_argument("--text", required=True)
    sp.add_argument("--batch", action="store_true")
    sp.set_defaults(func=cmd_embed)

    sp = sub.add_parser("ingest")
    sp.add_argument("--db", required=True)
    sp.add_argument("--embed-model", required=True)
    sp.add_argument("--max-chars", type=int, default=1200)
    sp.add_argument("--batch-size", type=int, default=32)
    sp.add_argument("files", nargs="+")
    sp.set_defaults(func=cmd_ingest)

    sp = sub.add_parser("rag")
    sp.add_argument("--db", required=True)
    sp.add_argument("--embed-model", required=True)
    sp.add_argument("--chat-model", required=True)
    sp.add_argument("--query", required=True)
    sp.add_argument("--top-k", type=int, default=6)
    sp.add_argument("--tools", action="store_true")
    sp.add_argument("--stream", action="store_true")
    sp.add_argument("--think", default=None)
    sp.add_argument("--keep-alive", dest="keep_alive", default=None)
    sp.add_argument("--max-steps", type=int, default=8)
    sp.add_argument("--format-json", action="store_true")
    sp.add_argument("--schema", default=None)
    sp.add_argument("--options-json", default=None)
    sp.add_argument("--print-thinking", action="store_true")
    sp.set_defaults(func=cmd_rag)

    sp = sub.add_parser("repl")
    sp.add_argument("--model", required=True)
    sp.add_argument("--system", default=None)
    sp.add_argument("--tools", action="store_true")
    sp.add_argument("--stream", action="store_true")
    sp.add_argument("--think", default=None)
    sp.add_argument("--keep-alive", dest="keep_alive", default=None)
    sp.add_argument("--max-steps", type=int, default=8)
    sp.add_argument("--format-json", action="store_true")
    sp.add_argument("--schema", default=None)
    sp.add_argument("--options-json", default=None)
    sp.add_argument("--print-thinking", action="store_true")
    sp.add_argument("--image", default=None)
    sp.set_defaults(func=cmd_repl)

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.cmd in ("repl", "rag"):
        if args.think is not None:
            val = str(args.think).strip()
            if val.lower() in ("true", "false"):
                args.think = (val.lower() == "true")
            else:
                args.think = val

    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())