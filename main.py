from __future__ import annotations

import json
import logging
import os
import time
import uuid
from typing import Any, AsyncGenerator, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

# ── Config ────────────────────────────────────────────────────────────────────

PORT = int(os.environ.get("PORT", 8000))
API_KEY = os.environ.get("API_KEY", "sk-qwen")  # set env var or leave default

# Cookies — override via env vars on Railway for easy rotation
COOKIES: dict[str, str] = {
    "acw_tc":           os.environ.get("QWEN_ACW_TC",   "0a03e59317847050140032723e2384ab466197beccb186a7241f653b630ba1"),
    "cna":              os.environ.get("QWEN_CNA",       "Pl7nIgmeFWwCAXBPCqRddPy/"),
    "sca":              os.environ.get("QWEN_SCA",       "02f5e288"),
    "atpsida":          os.environ.get("QWEN_ATPSIDA",   "68d3655c1c1265dafdedc90f_1784705527_11"),
    "x-ap":             "ap-southeast-1",
    "qwen-theme":       "light",
    "qwen-locale":      "en-US",
    "xlly_s":           "1",
    "isg":              os.environ.get("QWEN_ISG",       "BEBAMhHePmMHasJnGlfu0ZgwEs4SySST388dmLrRsdvuNehfZNvLIkBDTQX1ntxr"),
    "_c_WBKFRo":        os.environ.get("QWEN_C_WBKFRO",  "KikJSoIiTH01JYkrI2BeZWWgIyRpVERRTWUGpqzi"),
    "_nb_ioWEgULi":     "",
    "token":            os.environ.get("QWEN_TOKEN",     "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6IjVlMDA2NDY1LTEzODQtNDljMS1iNjBlLTUyMjg2MWEyNGY4OSIsImxhc3RfcGFzc3dvcmRfY2hhbmdlIjoxNzg0NzA1MzAwLCJleHAiOjE3ODcyOTc1MDh9.RD4lV8jgJ1ZlyXqJ2uxRVtx1cztAKzlzK1sENWESuU8"),
    "cnaui":            os.environ.get("QWEN_CNAUI",     "5e006465-1384-49c1-b60e-522861a24f89"),
    "aui":              os.environ.get("QWEN_AUI",       "5e006465-1384-49c1-b60e-522861a24f89"),
    "_gcl_au":          os.environ.get("QWEN_GCL_AU",    "1.1.1579056343.1784705405"),
    "qwen-thinking_mode": "Thinking",
}

HEADERS: dict[str, str] = {
    "User-Agent":    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:152.0) Gecko/20100101 Firefox/152.0",
    "Accept":        "text/event-stream",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer":       "https://chat.qwen.ai/",
    "Version":       "0.2.75",
    "source":        "web",
    "bx-umidtoken":  os.environ.get("QWEN_BX_UMID", "T2gArKD7nVt2YKHxk8PZpIrUSRbwyrwLXwqfHsY9VqpKxEvMBUx21KmQSG2ysJk3stM="),
    "bx-v":          "2.5.36",
    "Content-Type":  "application/json",
    "Connection":    "keep-alive",
}

CHAT_COMPLETIONS_URL = "https://chat.qwen.ai/api/chat/completions"

MODEL_MAP: dict[str, str] = {
    "qwen-max":    "qwen-max",
    "qwen-plus":   "qwen-plus",
    "qwen-turbo":  "qwen-turbo",
    "qwen-long":   "qwen-long",
    "qwen3-235b":  "qwen3-235b-a22b",
    "qwen3-32b":   "qwen3-32b",
    "qwq-32b":     "qwq-32b",
}
DEFAULT_MODEL = os.environ.get("QWEN_DEFAULT_MODEL", "qwen-max")

# Context window sizes — Zed uses context_length for tool/prompt planning
MODEL_CONTEXT: dict[str, int] = {
    "qwen-max":       32768,
    "qwen-plus":      131072,
    "qwen-turbo":     131072,
    "qwen-long":      1000000,
    "qwen3-235b":     131072,
    "qwen3-32b":      131072,
    "qwq-32b":        131072,
}

app = FastAPI(title="Qwen OpenAI-compatible API")
logging.basicConfig(level=logging.WARNING)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _flatten_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                t = block.get("type", "")
                if t == "text":
                    parts.append(block.get("text", ""))
                elif t == "tool_result":
                    parts.append(_flatten_content(block.get("content", "")))
                elif t not in ("image", "image_url"):
                    for key in ("text", "content", "value"):
                        v = block.get(key)
                        if isinstance(v, str) and v:
                            parts.append(v)
                            break
        return "\n".join(p for p in parts if p)
    if content is None:
        return ""
    return str(content)


def _resolve_model(model: str) -> str:
    return MODEL_MAP.get(model, model or DEFAULT_MODEL)


def _is_thinking_model(model: str) -> bool:
    return any(k in model.lower() for k in ("qwq", "think", "reason"))


def _process_chunk(data: dict, strip_thinking: bool) -> Optional[str]:
    choices = data.get("choices") or []
    if not choices:
        return None
    delta = choices[0].get("delta") or {}
    text = delta.get("content") or ""
    thinking = delta.get("reasoning_content") or ""

    if thinking and not strip_thinking:
        return thinking  # caller wraps in <think> tags at boundary

    return text if text else None


def _sse(cid: str, ts: int, model: str, delta: dict = {}, finish_reason: Optional[str] = None) -> str:
    payload = {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": ts,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(payload)}\n\n"


def _check_auth(request: Request) -> None:
    if not API_KEY:
        return
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer ") or auth[7:] != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


def _build_qwen_payload(messages: list[dict], model: str, chat_id: str) -> dict:
    return {
        "stream": True,
        "incremental_output": True,
        "chat_type": "t2t",
        "model": model,
        "messages": messages,
        "id": chat_id,
        "fid": "",
        "pid": "0",
    }


def _model_entry(mid: str) -> dict:
    """OpenAI-compatible model object with fields Zed Agent expects."""
    return {
        "id": mid,
        "object": "model",
        "created": 0,
        "owned_by": "qwen",
        # Zed reads these:
        "context_length": MODEL_CONTEXT.get(mid, 32768),
        "max_tokens": MODEL_CONTEXT.get(mid, 32768),
        "capabilities": {
            "completion": True,
            "chat_completion": True,
            "embeddings": False,
            "tool_choice": False,   # Qwen web API doesn't expose tool calls
        },
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "service": "qwen-bypass"}


@app.get("/v1/models")
def list_models(request: Request):
    _check_auth(request)
    return {
        "object": "list",
        "data": [_model_entry(m) for m in MODEL_MAP],
    }


@app.get("/v1/models/{model_id:path}")
def get_model(model_id: str, request: Request):
    _check_auth(request)
    resolved = _resolve_model(model_id)
    if resolved not in MODEL_MAP.values() and model_id not in MODEL_MAP:
        raise HTTPException(status_code=404, detail=f"Model {model_id!r} not found")
    return _model_entry(model_id if model_id in MODEL_MAP else resolved)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    _check_auth(request)

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    logging.warning("BODY: %s", json.dumps(body)[:800])

    raw_model  = body.get("model") or DEFAULT_MODEL
    qwen_model = _resolve_model(raw_model)
    stream     = bool(body.get("stream", False))
    strip_thinking = not _is_thinking_model(qwen_model)

    # Zed may send tool_choice / tools — we silently ignore them since
    # the Qwen web endpoint doesn't support native tool calls.
    # tool_choice = body.get("tool_choice")
    # tools       = body.get("tools")

    raw_messages = body.get("messages") or []
    messages: list[dict] = []
    for m in raw_messages:
        if not isinstance(m, dict):
            continue
        role    = str(m.get("role", "user"))
        content = _flatten_content(m.get("content", ""))
        # Zed sometimes sends tool messages; fold them as user context
        if role == "tool":
            role = "user"
            content = f"[tool result]\n{content}"
        messages.append({"role": role, "content": content})

    if not messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    chat_id     = str(uuid.uuid4())
    qwen_payload = _build_qwen_payload(messages, qwen_model, chat_id)
    cid          = f"chatcmpl-{uuid.uuid4().hex}"
    ts           = int(time.time())

    if stream:
        return StreamingResponse(
            _stream(qwen_payload, cid, ts, qwen_model, strip_thinking),
            media_type="text/event-stream",
            headers={
                # Required for Zed's SSE client
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )
    return await _collect(qwen_payload, cid, ts, qwen_model, strip_thinking)


# ── Streaming ─────────────────────────────────────────────────────────────────

async def _stream(
    payload: dict,
    cid: str,
    ts: int,
    model: str,
    strip_thinking: bool,
) -> AsyncGenerator[str, None]:
    yield _sse(cid, ts, model, delta={"role": "assistant", "content": ""})

    in_think = False  # track open <think> block for reasoning models

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream(
                "POST",
                CHAT_COMPLETIONS_URL,
                headers=HEADERS,
                cookies=COOKIES,
                json=payload,
            ) as resp:
                if resp.status_code != 200:
                    body_bytes = await resp.aread()
                    err = body_bytes.decode(errors="replace")[:200]
                    yield _sse(cid, ts, model, delta={"content": f"\n\n[Qwen error {resp.status_code}: {err}]"})
                    yield _sse(cid, ts, model, finish_reason="stop")
                    yield "data: [DONE]\n\n"
                    return

                async for raw_line in resp.aiter_lines():
                    if not raw_line:
                        continue
                    if not raw_line.startswith("data:"):
                        continue
                    chunk = raw_line[5:].strip()
                    if chunk == "[DONE]":
                        break
                    try:
                        data = json.loads(chunk)
                    except json.JSONDecodeError:
                        continue

                    choices = data.get("choices") or []
                    if not choices:
                        continue
                    delta    = choices[0].get("delta") or {}
                    text     = delta.get("content") or ""
                    thinking = delta.get("reasoning_content") or ""

                    if thinking and not strip_thinking:
                        if not in_think:
                            yield _sse(cid, ts, model, delta={"content": "<think>"})
                            in_think = True
                        yield _sse(cid, ts, model, delta={"content": thinking})
                    elif text:
                        if in_think:
                            yield _sse(cid, ts, model, delta={"content": "</think>\n"})
                            in_think = False
                        yield _sse(cid, ts, model, delta={"content": text})

    except Exception as exc:
        logging.exception("Stream error")
        if in_think:
            yield _sse(cid, ts, model, delta={"content": "</think>\n"})
        yield _sse(cid, ts, model, delta={"content": f"\n\n[Error: {exc}]"})

    if in_think:
        yield _sse(cid, ts, model, delta={"content": "</think>\n"})

    yield _sse(cid, ts, model, finish_reason="stop")
    yield "data: [DONE]\n\n"


# ── Non-streaming ─────────────────────────────────────────────────────────────

async def _collect(
    payload: dict,
    cid: str,
    ts: int,
    model: str,
    strip_thinking: bool,
) -> JSONResponse:
    full_text = ""
    in_think  = False

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream(
                "POST",
                CHAT_COMPLETIONS_URL,
                headers=HEADERS,
                cookies=COOKIES,
                json=payload,
            ) as resp:
                if resp.status_code != 200:
                    body_bytes = await resp.aread()
                    raise HTTPException(status_code=502, detail=body_bytes.decode(errors="replace")[:300])

                async for raw_line in resp.aiter_lines():
                    if not raw_line or not raw_line.startswith("data:"):
                        continue
                    chunk = raw_line[5:].strip()
                    if chunk == "[DONE]":
                        break
                    try:
                        data = json.loads(chunk)
                    except json.JSONDecodeError:
                        continue

                    choices  = data.get("choices") or []
                    if not choices:
                        continue
                    delta    = choices[0].get("delta") or {}
                    text     = delta.get("content") or ""
                    thinking = delta.get("reasoning_content") or ""

                    if thinking and not strip_thinking:
                        if not in_think:
                            full_text += "<think>"
                            in_think = True
                        full_text += thinking
                    elif text:
                        if in_think:
                            full_text += "</think>\n"
                            in_think = False
                        full_text += text

    except HTTPException:
        raise
    except Exception as exc:
        logging.exception("Collect error")
        raise HTTPException(status_code=502, detail=str(exc))

    if in_think:
        full_text += "</think>\n"

    return JSONResponse({
        "id": cid,
        "object": "chat.completion",
        "created": ts,
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": full_text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    })


# ── Entrypoint (Railway runs this directly via Procfile) ──────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, log_level="warning")
