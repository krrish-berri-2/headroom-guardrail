"""
Headroom prompt-compression guardrail microservice.

Implements the LiteLLM generic guardrail API spec so it can sit as a
pre_call guardrail that compresses incoming messages before they reach
the LLM provider.

Spec: https://docs.litellm.ai/docs/adding_provider/generic_guardrail_api
Headroom: https://headroom-docs.vercel.app/docs
"""

from __future__ import annotations

import logging
import os
from typing import Annotated, Any, Optional

from fastapi import FastAPI, Header, HTTPException, status
from headroom import compress
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

app = FastAPI(title="Headroom Guardrail", version="0.1.0")

_API_KEY = os.environ.get("GUARDRAIL_API_KEY")


class MessageDict(BaseModel):
    role: str
    content: Any


class GuardrailRequest(BaseModel):
    input_type: str
    texts: Optional[list[str]] = None
    images: Optional[list[str]] = None
    structured_messages: Optional[list[dict[str, Any]]] = None
    tools: Optional[list[dict[str, Any]]] = None
    tool_calls: Optional[list[dict[str, Any]]] = None
    model: Optional[str] = None
    litellm_call_id: Optional[str] = None
    litellm_trace_id: Optional[str] = None
    request_data: Optional[dict[str, Any]] = None
    additional_provider_specific_params: Optional[dict[str, Any]] = Field(
        default=None
    )


class GuardrailResponse(BaseModel):
    action: str
    texts: Optional[list[str]] = None
    images: Optional[list[str]] = None
    structured_messages: Optional[list[dict[str, Any]]] = None
    blocked_reason: Optional[str] = None


def _extract_texts_from_messages(messages: list[dict[str, Any]]) -> list[str]:
    texts: list[str] = []
    for msg in messages:
        content: Any = msg.get("content")
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            for part in content:  # type: ignore[union-attr]
                if isinstance(part, str):
                    texts.append(part)
                elif isinstance(part, dict):
                    part_dict: dict[str, Any] = part
                    if part_dict.get("type") == "text":
                        text = part_dict.get("text")
                        if isinstance(text, str):
                            texts.append(text)
    return texts


def _resolve_model(request: GuardrailRequest) -> str:
    if request.model:
        return request.model
    fallback = os.environ.get("HEADROOM_DEFAULT_MODEL", "gpt-4o-mini")
    return fallback


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


class DebugResponse(BaseModel):
    tokens_saved: Any
    compression_ratio: Any
    tokens_before: Any
    tokens_after: Any
    input_message_count: int
    output_message_count: int
    transforms_applied: Any
    input_texts: list[str]
    output_texts: list[str]


@app.post("/debug/compress", response_model=DebugResponse)
async def debug_compress(request: GuardrailRequest) -> DebugResponse:
    messages = request.structured_messages or []
    model = _resolve_model(request)
    result = compress(messages, model=model, compress_user_messages=True, protect_recent=0)
    compressed: list[dict[str, Any]] = result.messages  # type: ignore[attr-defined]
    return DebugResponse(
        tokens_saved=getattr(result, "tokens_saved", None),
        compression_ratio=getattr(result, "compression_ratio", None),
        tokens_before=getattr(result, "tokens_before", None),
        tokens_after=getattr(result, "tokens_after", None),
        input_message_count=len(messages),
        output_message_count=len(compressed),
        transforms_applied=getattr(result, "transforms_applied", []),
        input_texts=_extract_texts_from_messages(messages),
        output_texts=_extract_texts_from_messages(compressed),
    )


@app.post("/beta/litellm_basic_guardrail_api", response_model=GuardrailResponse)
async def guardrail(
    request: GuardrailRequest,
    x_api_key: Annotated[Optional[str], Header()] = None,
) -> GuardrailResponse:
    if _API_KEY and x_api_key != _API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")

    if request.input_type != "request":
        return GuardrailResponse(action="NONE")

    messages = request.structured_messages
    if not messages:
        return GuardrailResponse(action="NONE")

    model = _resolve_model(request)

    try:
        result = compress(messages, model=model, compress_user_messages=True, protect_recent=0)
    except Exception:
        logger.exception(
            "headroom compress failed (call_id=%s); passing through unchanged",
            request.litellm_call_id,
        )
        return GuardrailResponse(action="NONE")

    compressed_messages: list[dict[str, Any]] = result.messages  # type: ignore[attr-defined]

    tokens_saved = getattr(result, "tokens_saved", 0)
    logger.info(
        "headroom compressed call_id=%s model=%s tokens_saved=%s compression_ratio=%s",
        request.litellm_call_id,
        model,
        tokens_saved,
        getattr(result, "compression_ratio", "?"),
    )

    if not tokens_saved:
        return GuardrailResponse(action="NONE")

    return GuardrailResponse(
        action="GUARDRAIL_INTERVENED",
        structured_messages=compressed_messages,
    )
