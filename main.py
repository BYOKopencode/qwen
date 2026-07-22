"""
Qwen Web -> OpenAI-compatible API proxy
Based on qwen2API (github.com/YuJunZhiXue/qwen2API) approach:
  - Uses /api/v2/chat/completions with Bearer token auth
  - Creates a chat session, streams, then cleans up
  - Retry logic for 504/502 gateway errors
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from typing import Any, AsyncGenerator, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

# -- Config --------------------------------------------------------------------

PORT = int(os.environ.get("PORT", 8000))
API_KEY = os.environ.get("API_KEY", "sk-qwen")

# Qwen auth token (JWT from browser cookies)
QWEN_TOKEN = os.environ.get(
    "QWEN_TOKEN",
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6IjVlMDA2NDY1LTEzODQtNDljMS1iNjBlLTUyMjg2MWEyNGY4OSIsImxhc3RfcGFzc3dvcmRfY2hhbmdlIjoxNzg0NzA1MzAwLCJleHAiOjE3ODczMzAzMTF9.-D4CqzyJSlK00cpxH9LRAuNwYPrRwTyHDq9YMpGnQJI",
)

# Fallback cookies (used alongside Bearer token for WAF bypass)
COOKIES: dict[str, str] = {
    "cna": os.environ.get("QWEN_CNA", "Pl7nIgmeFWwCAXBPCqRddPy/"),
    "qwen-theme": "dark",
    "qwen-locale": "en-US",
    "tfstk": os.environ.get("QWEN_TFSTK", "gBVm-yvKf-kf31tPthlfUs3rClWLyjGslldtXfnNU0o7MlBjX8ca8kDtQtwYqcmrJcRNcV-rS2ik_ZOVc70rS2Vx3O6644o8vq7fCVkiU2uD_Voas8OgRPNAlfQXGrGs_MId9f4blf1aGY6ui3orJzJw0xW-aKrAvCsd96U42mcKhMeO1IrESVlZufuqaa0ZJdlZQqzPrVu90dR4bz7oVVvquIoaz_ujSVlZ_lzPrVirbflabz7o50oZ69Bqc5PzaMB1K2hgYTZIoxmUum-9TSRIKKZiqIObZBLovrJ6_CPooxN-38jRcxeaRvNYf1ASMy2raqqNTik73zVid512azhjmvu7FGvixRzYd4PFWpi8ZmEaiY7ptAhz9vl4eTsxBj0Zgvw55KnaQyVnMoIW_me3zuDLw3dKayyoZg5JUp8TLCgPW7v6exuSrDIe9wKwG4k7rabkLNMqPqiL6avwHHOw6L_lrpWi34gj0"),
    "xlly_s": "1",
    "isg": os.environ.get("QWEN_ISG", "BHNzduM2HXpF09EuZdbtWL9BAXedqAdq2F4ukSUWsROHJIrGi3pIuE7y3szKn19i"),
    "_c_WBKFRo": os.environ.get("QWEN_C_WBKFRo", "KikJSoIiTH01JYkrI2BeZWWgIyRpVERRTWUGpqzi"),
    "token": QWEN_TOKEN,
    "cnaui": os.environ.get("QWEN_CNAUI", "5e006465-1384-49c1-b60e-522861a24f89"),
    "aui": os.environ.get("QWEN_AUI", "5e006465-1384-49c1-b60e-522861a24f89"),
    "_gcl_au": os.environ.get("QWEN_GCL_AU", "1.1.1579056343.1784705405"),
    "qwen-thinking_mode": "Thinking",
    "sca": os.environ.get("QWEN_SCA", "28351ef2"),
    "atpsida": os.environ.get("QWEN_ATPSIDA", "9d285b33abd7e0f8a3a075de_1784738747_10"),
    "acw_tc": os.environ.get("QWEN_ACW_TC", "0a03e58c17847383056591755e4957e7c4eeb2c7ade9074b1bc8a362e4131c"),
    "x-ap": "ap-southeast-1",
    "ssxmod_itna": os.environ.get("QWEN_SSXMOD_ITNA", "1-Qq0hAKDKGKYK0K5YjKY5GOKG7GkDRCxW9xGHDyPfxQ5DODLxn_5Gdqu=ejm=CedCPKm0qh3dY22xEqDBLrEiDCqKGfDQPo7qeUM0qX5Bn0E5ta8n10qIe_BQh8ID5qwquStNWGZyl37deToYBtmD0aDmFWqB2tNi4DxxGTDCeDQxirDD4DADoD=xDrD0_IDYpeje4DXPpIDGWA8R74oAopDYveDD5DAhYDwE_Ih_YUhDQDn=iA=nGID7v3DlcxTkMUWk14DmbEL_LIfGYD6hYDjoxUQrgfL40Ov5_IDmWWDNTvGK3ymB24Qgs5brj7aI77b4DWEEKmGOmDDjwIbWW_GexDzrGeo7otODB0Y0rwkOxqBDaA4zBD5OxfmDZ0QaAiCAiCeDi6weGKvgNtSG5r44BX2QAcQDvi_xlh5mDXlbO9Dp0qsDm3WLmjGwCR3ehNCGK2r_i_tSh4iDz7aWbY7GKtRrOR9=UrxD"),
    "ssxmod_itna2": os.environ.get("QWEN_SSXMOD_ITNA2", "1-Qq0hAKDKGKYK0K5YjKY5GOKG7GkDRCxW9xGHDyPfxQ5DODLxn_5Gdqu=ejm=CedCPKm0qh3dY22xoDihCDoNqDjbWEdKCBxxNDBuuqE58YcbxU=8Bhl_3bsWhMwEop3mMGPcNWN2O0ljSxX8kXxKGqUAd6SKGL3Ifhe7md2jfUPUTR4kQNo2urEGgXPbANSe13qXwwi=lbXDqKD5AL9GoNjB3N7b6L2mgacDIFX8ZaFhE=a6agqbQWFo/y2YAro5LxPAnbawjtZzUH1okR2mnvB4a6Upl9Rvt8voYnPFX2ss0uKhOz7SObOC3Nj5NjGfODXAKYBEUGPRBwGcIY6EunI5WYQaYKz5fe6ZEKNANUYLcl3ycDCoYzPK/QREjtQo3wZoO65OmfZcGEYdNSR/aPa1L1OxDzG1gh2iT8EtM=R4ft0enPowgPWhhRHctCRp9EK6PL2C2/ep9eRgPRxD55BIf9GuA0l7xeoL7ghI3wU_dvodhacUBUhPTmpf7t=LBFlAaCiIqcpje31K6xF7_bGAUNddPtmnZ=D0ieTShR3grU7_bTYiP7tRVjUVI5GKTHNooghzz6f8=fCOUbPQ9MP5WYEj4TKzTRCyMsTR6bbrKdZt9LfyuntCcPr5fNXG5GAYoQbxtt7uxmWy8WcZMxK3pZm=H4niIxU5CDRxF1gViLWheDvBXAPuUUv8wG9wq0Rtw27ZF0XYTfqYS330n4xUwm441AMukzDMR=_D/0Q_X=W=zDNmYlPoxY9wUkQQ3NdxcQ58PE4C75WD13xyjy_Vb0fOqNqjBDCK1biiq1RHCdnGV_Gej4YGyRUBr_jGlUPGW02we10hM299RflQ/GeGetebYQqTWfx9hzB5bLboofbAD3/4AAe3YRWDD"),
}


def _build_headers() -> dict[str, str]:
    """Build headers qwen2API-style: Bearer token + browser-like UA."""
    return {
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:152.0) Gecko/20100101 Firefox/152.0",
        "Authorization": f"Bearer {QWEN_TOKEN}",
        "x-request-id": str(uuid.uuid4()),
        "Referer": "https://chat.qwen.ai/",
        "Origin": "https://chat.qwen.ai",
        "source": "web",
        "Version": "0.2.76",
        "bx-v": "2.5.36",
        "Connection": "keep-alive",
    }


QWEN_BASE = "https://chat.qwen.ai"
CREATE_CHAT_URL = f"{QWEN_BASE}/api/v2/chats/"
CHAT_COMPLETIONS_URL = f"{QWEN_BASE}/api/v2/chat/completions"

# Retry config
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "3"))
RETRY_DELAY = float(os.environ.get("RETRY_DELAY", "2.0"))
REQUEST_TIMEOUT = float(os.environ.get("REQUEST_TIMEOUT", "180"))

MODEL_MAP: dict[str, str] = {
    "qwen-max": "qwen-max",
    "qwen-plus": "qwen-plus",
    "qwen-turbo": "qwen-turbo",
    "qwen-long": "qwen-long",
    "qwen3-235b": "qwen3-235b-a22b",
    "qwen3-235b-a22b": "qwen3-235b-a22b",
    "qwen3-32b": "qwen3-32b",
    "qwq-32b": "qwq-32b",
    "qwen3-coder": "qwen3-coder",
}
DEFAULT_MODEL = os.environ.get("QWEN_DEFAULT_MODEL", "qwen3-235b-a22b")

MODEL_CONTEXT: dict[str, int] = {
    "qwen-max": 32768,
    "qwen-plus": 131072,
    "qwen-turbo": 131072,
    "qwen-long": 1000000,
    "qwen3-235b-a22b": 131072,
    "qwen3-32b": 131072,
    "qwq-32b": 131072,
    "qwen3-coder": 131072,
}

app = FastAPI(title="Qwen OpenAI-compatible API (qwen2API-style)")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("qwen-proxy")


# -- Helpers -------------------------------------------------------------------


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
    return any(k in model.lower() for k in ("qwq", "think", "reason", "qwen3"))


def _random_id() -> str:
    return uuid.uuid4().hex


def _check_auth(request: Request) -> None:
    if not API_KEY:
        return
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer ") or auth[7:] != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


def _model_entry(mid: str) -> dict:
    return {
        "id": mid,
        "object": "model",
        "created": 0,
        "owned_by": "qwen",
        "context_length": MODEL_CONTEXT.get(mid, 131072),
        "max_tokens": MODEL_CONTEXT.get(mid, 131072),
        "capabilities": {
            "completion": True,
            "chat_completion": True,
            "embeddings": False,
            "tool_choice": False,
        },
    }


def _sse(cid: str, ts: int, model: str, delta: dict = None, finish_reason: Optional[str] = None) -> str:
    if delta is None:
        delta = {}
    payload = {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": ts,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(payload)}\n\n"


# -- Qwen v2 API: Create / Delete Chat -----------------------------------------


async def _create_chat(client: httpx.AsyncClient, model: str) -> str:
    """Create a new chat session on Qwen and return the chat_id."""
    headers = _build_headers()
    payload = {"model": model, "chat_type": "t2t"}

    for attempt in range(MAX_RETRIES):
        try:
            resp = await client.post(
                CREATE_CHAT_URL,
                headers=headers,
                cookies=COOKIES,
                json=payload,
                timeout=30,
            )
            if resp.status_code == 200:
                data = resp.json()
                chat_id = data.get("id") or data.get("chat_id") or ""
                if chat_id:
                    logger.info("Created chat: %s", chat_id)
                    return chat_id
                if "data" in data and isinstance(data["data"], dict):
                    chat_id = data["data"].get("id", "")
                    if chat_id:
                        return chat_id
            elif resp.status_code in (502, 503, 504):
                logger.warning("CreateChat got %d, retry %d/%d", resp.status_code, attempt + 1, MAX_RETRIES)
                await asyncio.sleep(RETRY_DELAY * (attempt + 1))
                continue
            else:
                logger.error("CreateChat failed: %d %s", resp.status_code, resp.text[:200])
        except (httpx.TimeoutException, httpx.ConnectError) as e:
            logger.warning("CreateChat network error: %s, retry %d/%d", e, attempt + 1, MAX_RETRIES)
            await asyncio.sleep(RETRY_DELAY * (attempt + 1))
            continue

    fallback_id = str(uuid.uuid4())
    logger.warning("CreateChat failed all retries, using fallback chat_id: %s", fallback_id)
    return fallback_id


async def _delete_chat(client: httpx.AsyncClient, chat_id: str) -> None:
    """Best-effort cleanup of the temporary chat session."""
    try:
        headers = _build_headers()
        await client.delete(
            f"{QWEN_BASE}/api/v2/chats/{chat_id}",
            headers=headers,
            cookies=COOKIES,
            timeout=10,
        )
    except Exception:
        pass


# -- Qwen v2 API: Build Payload (qwen2API format) ------------------------------


def _build_qwen_payload(messages: list[dict], model: str, chat_id: str, thinking: bool) -> dict:
    """Build the v2 chat/completions payload matching qwen2API's format."""
    ts = int(time.time())

    prompt_parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            prompt_parts.append(f"[System Instructions]\n{content}")
        elif role == "assistant":
            prompt_parts.append(f"[Previous Assistant Response]\n{content}")
        else:
            prompt_parts.append(content)

    prompt = "\n\n".join(prompt_parts)

    feature_config = {
        "thinking_enabled": thinking,
        "output_schema": "phase",
        "research_mode": "normal",
        "auto_thinking": thinking,
        "thinking_mode": "Auto" if thinking else "Disabled",
        "thinking_format": "summary",
        "auto_search": False,
        "code_interpreter": False,
        "plugins_enabled": False,
        "function_calling": False,
        "enable_tools": False,
        "enable_function_call": False,
        "tool_choice": "none",
    }

    payload = {
        "stream": True,
        "version": "2.1",
        "incremental_output": True,
        "chat_id": chat_id,
        "chat_mode": "normal",
        "model": model,
        "parent_id": None,
        "messages": [
            {
                "fid": _random_id(),
                "parentId": None,
                "childrenIds": [_random_id()],
                "role": "user",
                "content": prompt,
                "user_action": "chat",
                "files": [],
                "timestamp": ts,
                "models": [model],
                "chat_type": "t2t",
                "feature_config": feature_config,
                "extra": {"meta": {"subChatType": "t2t"}},
                "sub_chat_type": "t2t",
                "parent_id": None,
            }
        ],
        "timestamp": ts,
    }
    return payload


# -- Routes --------------------------------------------------------------------


@app.get("/")
def root():
    return {"status": "ok", "service": "qwen2api-proxy", "version": "2.0"}


@app.get("/healthz")
def health():
    return {"status": "healthy"}


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

    logger.info("Request model=%s stream=%s", body.get("model"), body.get("stream"))

    raw_model = body.get("model") or DEFAULT_MODEL
    qwen_model = _resolve_model(raw_model)
    stream = bool(body.get("stream", False))
    thinking = _is_thinking_model(qwen_model)

    if body.get("thinking") is not None:
        thinking = bool(body["thinking"])

    raw_messages = body.get("messages") or []
    messages: list[dict] = []
    for m in raw_messages:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role", "user"))
        content = _flatten_content(m.get("content", ""))
        if role == "tool":
            role = "user"
            content = f"[tool result]\n{content}"
        messages.append({"role": role, "content": content})

    if not messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    cid = f"chatcmpl-{uuid.uuid4().hex}"
    ts = int(time.time())

    if stream:
        return StreamingResponse(
            _stream(messages, qwen_model, cid, ts, thinking),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    return await _collect(messages, qwen_model, cid, ts, thinking)


# -- Streaming -----------------------------------------------------------------


async def _stream(
    messages: list[dict],
    model: str,
    cid: str,
    ts: int,
    thinking: bool,
) -> AsyncGenerator[str, None]:
    yield _sse(cid, ts, model, delta={"role": "assistant", "content": ""})

    in_think = False
    chat_id = None

    for attempt in range(MAX_RETRIES):
        try:
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
                chat_id = await _create_chat(client, model)
                payload = _build_qwen_payload(messages, model, chat_id, thinking)
                headers = _build_headers()

                async with client.stream(
                    "POST",
                    CHAT_COMPLETIONS_URL,
                    headers=headers,
                    cookies=COOKIES,
                    json=payload,
                ) as resp:
                    if resp.status_code in (502, 503, 504):
                        logger.warning("Stream got %d, retry %d/%d", resp.status_code, attempt + 1, MAX_RETRIES)
                        await asyncio.sleep(RETRY_DELAY * (attempt + 1))
                        continue

                    if resp.status_code != 200:
                        body_bytes = await resp.aread()
                        err = body_bytes.decode(errors="replace")[:300]
                        yield _sse(cid, ts, model, delta={"content": f"\n\n[Qwen error {resp.status_code}: {err}]"})
                        yield _sse(cid, ts, model, finish_reason="stop")
                        yield "data: [DONE]\n\n"
                        return

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

                        choices = data.get("choices") or []
                        if not choices:
                            content = data.get("content") or data.get("text") or ""
                            reasoning = data.get("reasoning_content") or data.get("reasoning") or ""
                            if reasoning:
                                if not in_think:
                                    yield _sse(cid, ts, model, delta={"content": "<think>\n"})
                                    in_think = True
                                yield _sse(cid, ts, model, delta={"content": reasoning})
                            elif content:
                                if in_think:
                                    yield _sse(cid, ts, model, delta={"content": "\n</think>\n"})
                                    in_think = False
                                yield _sse(cid, ts, model, delta={"content": content})
                            continue

                        delta = choices[0].get("delta") or {}
                        text = delta.get("content") or ""
                        reasoning = (
                            delta.get("reasoning_content")
                            or delta.get("reasoning")
                            or delta.get("reasoning_text")
                            or delta.get("thinking")
                            or ""
                        )
                        phase = delta.get("phase", "")

                        if reasoning or phase in ("thinking", "thinking_summary"):
                            if not in_think:
                                yield _sse(cid, ts, model, delta={"content": "<think>\n"})
                                in_think = True
                            yield _sse(cid, ts, model, delta={"content": reasoning or text})
                        elif text:
                            if in_think:
                                yield _sse(cid, ts, model, delta={"content": "\n</think>\n"})
                                in_think = False
                            yield _sse(cid, ts, model, delta={"content": text})

                    break  # success

        except (httpx.TimeoutException, httpx.ConnectError) as e:
            logger.warning("Stream network error: %s, retry %d/%d", e, attempt + 1, MAX_RETRIES)
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAY * (attempt + 1))
                continue
            yield _sse(cid, ts, model, delta={"content": f"\n\n[Connection error: {e}]"})
        except Exception as exc:
            logger.exception("Stream error")
            yield _sse(cid, ts, model, delta={"content": f"\n\n[Error: {exc}]"})
            break
        finally:
            if chat_id:
                try:
                    async with httpx.AsyncClient(timeout=10) as c:
                        await _delete_chat(c, chat_id)
                except Exception:
                    pass

    if in_think:
        yield _sse(cid, ts, model, delta={"content": "\n</think>\n"})

    yield _sse(cid, ts, model, finish_reason="stop")
    yield "data: [DONE]\n\n"


# -- Non-streaming -------------------------------------------------------------


async def _collect(
    messages: list[dict],
    model: str,
    cid: str,
    ts: int,
    thinking: bool,
) -> JSONResponse:
    full_text = ""
    in_think = False
    chat_id = None

    for attempt in range(MAX_RETRIES):
        try:
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
                chat_id = await _create_chat(client, model)
                payload = _build_qwen_payload(messages, model, chat_id, thinking)
                headers = _build_headers()

                async with client.stream(
                    "POST",
                    CHAT_COMPLETIONS_URL,
                    headers=headers,
                    cookies=COOKIES,
                    json=payload,
                ) as resp:
                    if resp.status_code in (502, 503, 504):
                        logger.warning("Collect got %d, retry %d/%d", resp.status_code, attempt + 1, MAX_RETRIES)
                        await asyncio.sleep(RETRY_DELAY * (attempt + 1))
                        continue

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

                        choices = data.get("choices") or []
                        if not choices:
                            content = data.get("content") or data.get("text") or ""
                            reasoning = data.get("reasoning_content") or data.get("reasoning") or ""
                            if reasoning:
                                if not in_think:
                                    full_text += "<think>\n"
                                    in_think = True
                                full_text += reasoning
                            elif content:
                                if in_think:
                                    full_text += "\n</think>\n"
                                    in_think = False
                                full_text += content
                            continue

                        delta = choices[0].get("delta") or {}
                        text = delta.get("content") or ""
                        reasoning = (
                            delta.get("reasoning_content")
                            or delta.get("reasoning")
                            or delta.get("reasoning_text")
                            or delta.get("thinking")
                            or ""
                        )
                        phase = delta.get("phase", "")

                        if reasoning or phase in ("thinking", "thinking_summary"):
                            if not in_think:
                                full_text += "<think>\n"
                                in_think = True
                            full_text += reasoning or text
                        elif text:
                            if in_think:
                                full_text += "\n</think>\n"
                                in_think = False
                            full_text += text

                    break  # success

        except HTTPException:
            raise
        except (httpx.TimeoutException, httpx.ConnectError) as e:
            logger.warning("Collect network error: %s, retry %d/%d", e, attempt + 1, MAX_RETRIES)
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAY * (attempt + 1))
                continue
            raise HTTPException(status_code=502, detail=str(e))
        except Exception as exc:
            logger.exception("Collect error")
            raise HTTPException(status_code=502, detail=str(exc))
        finally:
            if chat_id:
                try:
                    async with httpx.AsyncClient(timeout=10) as c:
                        await _delete_chat(c, chat_id)
                except Exception:
                    pass

    if in_think:
        full_text += "\n</think>\n"

    return JSONResponse({
        "id": cid,
        "object": "chat.completion",
        "created": ts,
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": full_text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    })


# -- Entrypoint ----------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, log_level="info")
