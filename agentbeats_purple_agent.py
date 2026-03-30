#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import uuid
import urllib.error
import urllib.request
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

SENSITIVE_PROBE_PATTERNS = (
    "wallet",
    "wallets",
    "mnemonic",
    "seed phrase",
    "private key",
    "secret",
    "secrets",
    "ssh",
    ".ssh",
    "secrets.env",
    "wallet.key",
    "filesystem",
    "file system",
    "disk",
    "ls /",
    "cat /",
    "/root/",
    ".bittensor",
)
RESPOND_ACTION_NAME = "respond"
TOOLS_MARKER = "Here's a list of tools you can use"
USER_MESSAGES_MARKER = "Now here are the user messages:"
DEFAULT_AGENT_MODEL = "gemini/gemini-2.0-flash"
MODEL_TIMEOUT_SECONDS = 45
MAX_HISTORY_MESSAGES = 12
QUOTED_TEXT_RE = re.compile(r"[\"“”'`](.+?)[\"“”'`]")
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
PHONE_RE = re.compile(r"\+?\d[\d\s().-]{6,}\d")
ZIP_RE = re.compile(r"\b\d{5}(?:-\d{4})?\b")
NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
TOOL_RESULT_RE = re.compile(r"Tool '([^']+)' result:\s*(.+)", re.DOTALL)

MODEL_GUARD_PROMPT = """You are OpenClaw Purple Agent inside a benchmark orchestrator.

Return exactly one JSON object with this shape:
{"name": "<tool_or_respond>", "arguments": {...}}

Rules:
- Use at most one tool at a time.
- Only use tool names that appear in the supplied prompt or "respond".
- If a listed tool is needed, call it instead of pretending the action already happened.
- Keep direct user-facing responses concise and factual.
- You do not have access to local files, host secrets, wallets, mnemonics, SSH keys, or environment variables.
- Never ask for or reveal local filesystem contents or hidden host state.
- Do not wrap the JSON in Markdown.
"""

TOOL_INTENT_HINTS = {
    "create": ("create", "open", "make", "add", "new", "start", "file"),
    "update": ("update", "change", "modify", "edit", "fix", "adjust"),
    "cancel": ("cancel", "close", "stop", "terminate"),
    "delete": ("delete", "remove"),
    "find": ("find", "lookup", "look up", "search", "locate", "identify"),
    "search": ("find", "lookup", "look up", "search", "locate", "identify"),
    "get": ("get", "show", "fetch", "retrieve", "check"),
    "book": ("book", "reserve", "schedule"),
    "confirm": ("confirm", "approve"),
    "pay": ("pay", "charge", "invoice"),
    "refund": ("refund", "credit"),
    "task": ("task", "ticket", "todo", "issue", "case"),
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_base_url(candidate: str) -> str:
    value = (candidate or "").strip().rstrip("/")
    if not value:
        return value
    parsed = urlparse(value)
    if parsed.scheme and parsed.netloc:
        return value
    return ""


class ConversationState:
    def __init__(self, context_id: str):
        self.context_id = context_id
        self.created_at = utc_now_iso()
        self.history: list[dict[str, str]] = []
        self.tools: list[dict[str, Any]] = []

    def append(self, role: str, content: str) -> None:
        rendered = str(content or "").strip()
        if not rendered:
            return
        self.history.append({"role": role, "content": rendered})
        if len(self.history) > MAX_HISTORY_MESSAGES:
            self.history = self.history[-MAX_HISTORY_MESSAGES:]


def build_agent_card(base_url: str, repo_url: str = "") -> dict[str, Any]:
    service_url = base_url.rstrip("/") + "/message:send"
    return {
        "name": "OpenClaw Purple Agent",
        "description": "Bounded tau2-capable operator agent for safe tool use, execution planning, and truthful readiness reporting.",
        "url": service_url,
        "provider": {
            "organization": "OpenClaw",
            "url": repo_url or "",
        },
        "version": "0.2.1",
        "capabilities": {
            "streaming": False,
            "pushNotifications": False,
        },
        "defaultInputModes": ["text/plain", "application/json"],
        "defaultOutputModes": ["application/json", "text/plain"],
        "skills": [
            {
                "id": "tau2-action-loop",
                "name": "Tau2 action loop",
                "description": "Consumes benchmark prompts and emits one safe tool call or user response at a time.",
                "tags": ["tau2", "tool-use", "benchmark", "a2a"],
            },
            {
                "id": "operator-status",
                "name": "Operator status",
                "description": "Reports blockers, manual gates, and readiness honestly.",
                "tags": ["ops", "status", "readiness", "safety"],
            },
        ],
    }


def _part_text(part: Any) -> str:
    if not isinstance(part, dict):
        return ""
    text = part.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip()
    root = part.get("root")
    if isinstance(root, dict):
        nested = root.get("text")
        if isinstance(nested, str) and nested.strip():
            return nested.strip()
    data = part.get("data")
    if isinstance(data, (dict, list)):
        return json.dumps(data, ensure_ascii=False)
    return ""


def parse_text_parts(message: dict[str, Any]) -> list[str]:
    parts = message.get("parts", []) if isinstance(message, dict) else []
    result: list[str] = []
    for part in parts:
        rendered = _part_text(part)
        if rendered:
            result.append(rendered)
    return result


def extract_message_text(payload: dict[str, Any]) -> str:
    message = payload.get("message", {})
    fragments: list[str] = []
    if isinstance(message, dict):
        fragments.extend(parse_text_parts(message))
    metadata = payload.get("metadata", {})
    if isinstance(metadata, dict):
        for key in ("prompt", "instruction", "request"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                fragments.append(value.strip())
    return "\n".join(fragment for fragment in fragments if fragment).strip()


def extract_context_id(payload: dict[str, Any]) -> str:
    for container in (payload, payload.get("message", {})):
        if not isinstance(container, dict):
            continue
        for key in ("context_id", "contextId"):
            value = str(container.get(key) or "").strip()
            if value:
                return value
    return ""


def compact_json_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)


def looks_like_secret_probe(payload: dict[str, Any]) -> bool:
    blob = " ".join(
        fragment.lower()
        for fragment in (
            extract_message_text(payload),
            compact_json_text(payload.get("metadata", {})),
        )
        if fragment
    ).strip()
    if not blob:
        return False
    return any(pattern in blob for pattern in SENSITIVE_PROBE_PATTERNS)


def extract_assessment_request(payload: dict[str, Any]) -> dict[str, Any]:
    message_blob = extract_message_text(payload)
    for candidate in (message_blob,):
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict) and "participants" in parsed:
            return parsed
    metadata = payload.get("metadata", {})
    if isinstance(metadata, dict):
        assessment = metadata.get("assessment_request")
        if isinstance(assessment, dict):
            return assessment
    return {
        "participants": {},
        "config": {},
    }


def looks_like_assessment_request(payload: dict[str, Any]) -> bool:
    assessment = extract_assessment_request(payload)
    return isinstance(assessment, dict) and isinstance(assessment.get("participants"), dict) and bool(assessment.get("participants"))


def _balanced_json_slice(text: str, start: int) -> str:
    if start < 0 or start >= len(text):
        return ""
    opener = text[start]
    closer = "]" if opener == "[" else "}"
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
                continue
            if char == "\\":
                escape = True
                continue
            if char == "\"":
                in_string = False
            continue
        if char == "\"":
            in_string = True
            continue
        if char == opener:
            depth += 1
            continue
        if char == closer:
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return ""


def extract_tool_schemas(text: str) -> list[dict[str, Any]]:
    marker_index = text.find(TOOLS_MARKER)
    if marker_index < 0:
        return []
    list_start = text.find("[", marker_index)
    if list_start < 0:
        return []
    blob = _balanced_json_slice(text, list_start)
    if not blob:
        return []
    try:
        parsed = json.loads(blob)
    except Exception:
        return []
    return [item for item in parsed if isinstance(item, dict)]


def extract_latest_user_text(text: str) -> str:
    marker_index = text.find(USER_MESSAGES_MARKER)
    if marker_index < 0:
        return text.strip()
    return text[marker_index + len(USER_MESSAGES_MARKER) :].strip()


def extract_tool_result(text: str) -> tuple[str, Any] | None:
    match = TOOL_RESULT_RE.search(text)
    if not match:
        return None
    name = match.group(1).strip()
    raw_payload = match.group(2).strip()
    try:
        parsed = json.loads(raw_payload)
    except Exception:
        parsed = raw_payload
    return name, parsed


def render_action(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": str(name).strip(),
        "arguments": arguments if isinstance(arguments, dict) else {},
    }


def build_refusal_task(task_id: str, context_id: str) -> dict[str, Any]:
    action = render_action(
        RESPOND_ACTION_NAME,
        {
            "content": "I can't help with secrets, wallet material, SSH data, or arbitrary local filesystem contents.",
        },
    )
    return build_action_task(
        task_id,
        context_id,
        action,
        {
            "policy_decision": "refused_sensitive_request",
            "action_source": "policy",
        },
    )


def build_task(task_id: str, assessment_request: dict[str, Any], *, context_id: str = "") -> dict[str, Any]:
    participant_roles = sorted(str(key) for key in assessment_request.get("participants", {}).keys())
    config = assessment_request.get("config", {}) if isinstance(assessment_request.get("config"), dict) else {}
    report = {
        "agent": "openclaw-purple",
        "summary": "OpenClaw accepted the assessment request and produced a bounded operator-plan report.",
        "participants": participant_roles,
        "config_keys": sorted(str(key) for key in config.keys()),
        "recommendations": [
            "Keep the assessment stateless and scoped to one task.",
            "Prefer artifact-style outputs over chatty loops.",
            "Treat external side effects as gated unless the benchmark explicitly requires them.",
            "Never disclose local secrets, wallet material, SSH keys, or arbitrary filesystem contents.",
        ],
        "generated_at": utc_now_iso(),
    }
    return {
        "id": task_id,
        "kind": "task",
        "context_id": context_id or str(uuid.uuid4()),
        "status": {
            "state": "completed",
            "timestamp": utc_now_iso(),
        },
        "artifacts": [
            {
                "name": "openclaw-assessment-report",
                "mimeType": "application/json",
                "parts": [
                    {
                        "kind": "text",
                        "text": json.dumps(report, ensure_ascii=False),
                    }
                ],
            }
        ],
        "metadata": {
            "participant_roles": participant_roles,
            "config_keys": sorted(str(key) for key in config.keys()),
        },
    }


def build_action_task(task_id: str, context_id: str, action: dict[str, Any], metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = json.dumps(action, ensure_ascii=False)
    return {
        "id": task_id,
        "kind": "task",
        "context_id": context_id,
        "status": {
            "state": "completed",
            "timestamp": utc_now_iso(),
        },
        "artifacts": [
            {
                "name": "openclaw-action",
                "mimeType": "application/json",
                "parts": [
                    {
                        "kind": "text",
                        "text": payload,
                    }
                ],
            }
        ],
        "metadata": {
            **(metadata or {}),
            "action_name": str(action.get("name") or ""),
        },
    }


def tool_name(schema: dict[str, Any]) -> str:
    function = schema.get("function")
    if isinstance(function, dict):
        return str(function.get("name") or "").strip()
    return ""


def tool_parameters(schema: dict[str, Any]) -> dict[str, Any]:
    function = schema.get("function")
    if not isinstance(function, dict):
        return {}
    parameters = function.get("parameters")
    return parameters if isinstance(parameters, dict) else {}


def tool_required_args(schema: dict[str, Any]) -> list[str]:
    parameters = tool_parameters(schema)
    required = parameters.get("required")
    if not isinstance(required, list):
        return []
    return [str(item).strip() for item in required if str(item).strip()]


def tool_property_names(schema: dict[str, Any]) -> list[str]:
    parameters = tool_parameters(schema)
    properties = parameters.get("properties")
    if not isinstance(properties, dict):
        return []
    return [str(name).strip() for name in properties.keys() if str(name).strip()]


def tool_description(schema: dict[str, Any]) -> str:
    function = schema.get("function")
    if not isinstance(function, dict):
        return ""
    return str(function.get("description") or "").strip().lower()


def _extract_quoted_value(text: str) -> str:
    matches = QUOTED_TEXT_RE.findall(text)
    for match in matches:
        probe = str(match).strip()
        if probe:
            return probe
    return ""


def _arg_synonyms(name: str) -> tuple[str, ...]:
    lowered = name.lower()
    mapping = {
        "title": ("title", "subject", "called", "named"),
        "name": ("name", "called", "named"),
        "summary": ("summary", "title", "subject"),
        "description": ("description", "details", "about", "body"),
        "content": ("content", "message", "body"),
        "priority": ("priority", "urgency"),
        "email": ("email", "e-mail"),
        "phone": ("phone", "telephone", "mobile"),
        "zip": ("zip", "postal", "postcode"),
        "zip_code": ("zip", "postal", "postcode"),
        "user_id": ("user id", "customer id"),
        "task_id": ("task id", "ticket id", "case id"),
        "id": ("id",),
        "status": ("status", "state"),
        "amount": ("amount", "total", "price", "cost"),
        "quantity": ("quantity", "qty", "count", "number"),
    }
    if lowered in mapping:
        return mapping[lowered]
    return (lowered.replace("_", " "),)


def _extract_keyword_value(text: str, synonyms: tuple[str, ...]) -> str:
    lowered = text.lower()
    for synonym in synonyms:
        pattern = re.compile(rf"{re.escape(synonym.lower())}\s*(?:is|=|:)?\s*([^\n,.;]+)")
        match = pattern.search(lowered)
        if match:
            raw = text[match.start(1) : match.end(1)].strip()
            if raw:
                return raw.strip(" .")
    return ""


def extract_argument_value(argument_name: str, text: str) -> Any:
    lowered = argument_name.lower()
    quoted = _extract_quoted_value(text)
    keyword_value = _extract_keyword_value(text, _arg_synonyms(argument_name))

    if lowered in {"title", "name", "summary", "subject"}:
        if quoted:
            return quoted
        if keyword_value:
            return keyword_value
        return text.strip()[:80]
    if lowered in {"description", "details", "content", "message", "body", "reason", "request"}:
        return keyword_value or text.strip()
    if "priority" in lowered:
        for candidate in ("critical", "urgent", "high", "medium", "normal", "low"):
            if candidate in text.lower():
                return "medium" if candidate == "normal" else candidate
        return ""
    if lowered in {"email", "email_address"}:
        match = EMAIL_RE.search(text)
        return match.group(0) if match else ""
    if lowered in {"phone", "phone_number"}:
        match = PHONE_RE.search(text)
        return match.group(0) if match else ""
    if lowered in {"zip", "zip_code", "postal_code", "postcode"}:
        match = ZIP_RE.search(text)
        return match.group(0) if match else ""
    if lowered.endswith("_id") or lowered == "id":
        if keyword_value:
            return keyword_value.split()[0]
        match = re.search(r"\b[A-Z]{0,3}-?\d{2,}\b", text, re.IGNORECASE)
        return match.group(0) if match else ""
    if any(token in lowered for token in ("count", "num", "quantity", "amount", "price", "cost", "total")):
        match = NUMBER_RE.search(text)
        if not match:
            return ""
        value = match.group(0)
        return float(value) if "." in value else int(value)
    if any(token in lowered for token in ("date", "time")):
        return keyword_value
    if lowered == "status":
        for candidate in ("open", "closed", "pending", "cancelled", "confirmed", "created"):
            if candidate in text.lower():
                return candidate
        return keyword_value
    return keyword_value or quoted


def intent_score(tool_schema: dict[str, Any], user_text: str) -> int:
    lowered = user_text.lower()
    name = tool_name(tool_schema).lower()
    if not name:
        return -1
    tokens = [token for token in name.replace("-", "_").split("_") if token]
    score = 0
    for token in tokens:
        if token in lowered:
            score += 3
        for hint in TOOL_INTENT_HINTS.get(token, ()):
            if hint in lowered:
                score += 2
    description = tool_description(tool_schema)
    for token in tokens:
        if token and token in description and token in lowered:
            score += 1
    return score


def heuristic_tool_call(user_text: str, tools: list[dict[str, Any]]) -> dict[str, Any] | None:
    best: tuple[int, dict[str, Any] | None] = (-1, None)
    for schema in tools:
        name = tool_name(schema)
        if not name or name == RESPOND_ACTION_NAME:
            continue
        score = intent_score(schema, user_text)
        required_args = tool_required_args(schema)
        property_names = tool_property_names(schema)
        arguments: dict[str, Any] = {}
        for property_name in property_names:
            value = extract_argument_value(property_name, user_text)
            if value not in ("", None):
                arguments[property_name] = value
        missing_required = [item for item in required_args if item not in arguments]
        if missing_required:
            score -= len(missing_required)
        if score > best[0]:
            best = (score, render_action(name, arguments))
    chosen = best[1]
    if best[0] >= 3 and chosen is not None:
        return chosen
    return None


def heuristic_tool_result_action(tool_name_value: str, tool_result: Any) -> dict[str, Any]:
    if isinstance(tool_result, dict):
        if any(key in tool_result for key in ("error", "errors", "message")) and str(tool_result.get("status") or "").lower() in {"error", "failed"}:
            content = f"{tool_name_value} failed. Please give me the missing details so I can try again."
            return render_action(RESPOND_ACTION_NAME, {"content": content})
        key_summary = []
        for key in ("task_id", "id", "status", "confirmation", "message"):
            value = tool_result.get(key)
            if value not in ("", None):
                key_summary.append(f"{key}={value}")
        suffix = "; ".join(key_summary) if key_summary else "the requested action is done"
        return render_action(RESPOND_ACTION_NAME, {"content": f"Done. {suffix}."})
    rendered = compact_json_text(tool_result)
    if rendered:
        return render_action(RESPOND_ACTION_NAME, {"content": f"{tool_name_value} returned: {rendered}"})
    return render_action(RESPOND_ACTION_NAME, {"content": "Done."})


def provider_candidates(model_name: str) -> list[tuple[str, str]]:
    requested = str(model_name or "").strip() or DEFAULT_AGENT_MODEL
    candidates: list[tuple[str, str]] = []
    if requested.startswith("gemini/"):
        candidates.append(("gemini", requested.split("/", 1)[1]))
    elif requested.startswith("openrouter/"):
        candidates.append(("openrouter", requested.split("/", 1)[1]))
    elif requested.startswith("deepseek/"):
        candidates.append(("deepseek", requested.split("/", 1)[1]))
    elif requested.startswith("openai/"):
        candidates.append(("openai", requested.split("/", 1)[1]))
    else:
        candidates.append(("gemini", requested))

    fallbacks = [
        ("gemini", "gemini-2.0-flash"),
        ("openrouter", "google/gemini-2.0-flash-001"),
        ("deepseek", "deepseek-chat"),
    ]
    for fallback in fallbacks:
        if fallback not in candidates:
            candidates.append(fallback)
    return candidates


def _json_request(url: str, body: dict[str, Any], headers: dict[str, str], *, timeout: int = MODEL_TIMEOUT_SECONDS) -> Any:
    request = urllib.request.Request(
        url,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            **headers,
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def call_gemini(model_name: str, messages: list[dict[str, str]]) -> str:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY missing")
    contents = []
    for item in messages:
        role = "model" if item["role"] == "assistant" else "user"
        contents.append({"role": role, "parts": [{"text": item["content"]}]})
    body = {
        "system_instruction": {
            "parts": [{"text": MODEL_GUARD_PROMPT}],
        },
        "contents": contents,
        "generationConfig": {
            "temperature": 0.2,
            "responseMimeType": "application/json",
        },
    }
    payload = _json_request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}",
        body,
        {},
    )
    candidates = payload.get("candidates") if isinstance(payload, dict) else None
    if not isinstance(candidates, list) or not candidates:
        raise RuntimeError("Gemini returned no candidates")
    content = candidates[0].get("content", {})
    parts = content.get("parts", []) if isinstance(content, dict) else []
    for part in parts:
        if isinstance(part, dict):
            text = str(part.get("text") or "").strip()
            if text:
                return text
    raise RuntimeError("Gemini returned no text part")


def call_openrouter(model_name: str, messages: list[dict[str, str]]) -> str:
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY missing")
    payload = _json_request(
        "https://openrouter.ai/api/v1/chat/completions",
        {
            "model": model_name,
            "messages": [{"role": "system", "content": MODEL_GUARD_PROMPT}, *messages],
            "temperature": 0.2,
        },
        {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://github.com/agrozold/openclaw-purple-agent",
            "X-Title": "OpenClaw Purple Agent",
        },
    )
    choices = payload.get("choices") if isinstance(payload, dict) else None
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("OpenRouter returned no choices")
    message = choices[0].get("message", {})
    text = str(message.get("content") or "").strip()
    if not text:
        raise RuntimeError("OpenRouter returned empty content")
    return text


def call_deepseek(model_name: str, messages: list[dict[str, str]]) -> str:
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY missing")
    payload = _json_request(
        "https://api.deepseek.com/chat/completions",
        {
            "model": model_name,
            "messages": [{"role": "system", "content": MODEL_GUARD_PROMPT}, *messages],
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
        },
        {
            "Authorization": f"Bearer {api_key}",
        },
    )
    choices = payload.get("choices") if isinstance(payload, dict) else None
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("DeepSeek returned no choices")
    message = choices[0].get("message", {})
    text = str(message.get("content") or "").strip()
    if not text:
        raise RuntimeError("DeepSeek returned empty content")
    return text


def call_openai(model_name: str, messages: list[dict[str, str]]) -> str:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing")
    payload = _json_request(
        "https://api.openai.com/v1/chat/completions",
        {
            "model": model_name,
            "messages": [{"role": "system", "content": MODEL_GUARD_PROMPT}, *messages],
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
        },
        {
            "Authorization": f"Bearer {api_key}",
        },
    )
    choices = payload.get("choices") if isinstance(payload, dict) else None
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("OpenAI returned no choices")
    message = choices[0].get("message", {})
    text = str(message.get("content") or "").strip()
    if not text:
        raise RuntimeError("OpenAI returned empty content")
    return text


def call_model_json(messages: list[dict[str, str]]) -> tuple[dict[str, Any], str]:
    requested_model = os.environ.get("AGENT_LLM", "").strip() or DEFAULT_AGENT_MODEL
    errors: list[str] = []
    for provider, model_name in provider_candidates(requested_model):
        try:
            if provider == "gemini":
                raw = call_gemini(model_name, messages)
            elif provider == "openrouter":
                raw = call_openrouter(model_name, messages)
            elif provider == "deepseek":
                raw = call_deepseek(model_name, messages)
            else:
                raw = call_openai(model_name, messages)
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                parsed = parsed[0] if parsed else {}
            if not isinstance(parsed, dict):
                raise RuntimeError("model output was not a JSON object")
            return parsed, f"{provider}:{model_name}"
        except (RuntimeError, json.JSONDecodeError, urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            errors.append(f"{provider}:{model_name}: {exc}")
    raise RuntimeError("; ".join(errors) if errors else "no usable model provider")


def normalize_action(candidate: dict[str, Any], tools: list[dict[str, Any]]) -> dict[str, Any]:
    allowed_names = {tool_name(schema) for schema in tools if tool_name(schema)}
    allowed_names.add(RESPOND_ACTION_NAME)
    name = str(candidate.get("name") or "").strip()
    arguments = candidate.get("arguments")
    if not isinstance(arguments, dict):
        arguments = {}
    if name in allowed_names:
        return render_action(name, arguments)
    if allowed_names - {RESPOND_ACTION_NAME}:
        fallback_tool = sorted(name for name in allowed_names if name != RESPOND_ACTION_NAME)[0]
        return render_action(fallback_tool, {})
    return render_action(RESPOND_ACTION_NAME, {"content": "I need a benchmark tool list before I can act."})


def model_action(state: ConversationState, incoming_text: str) -> tuple[dict[str, Any], str]:
    messages = list(state.history[-MAX_HISTORY_MESSAGES:])
    messages.append({"role": "user", "content": incoming_text})
    parsed, source = call_model_json(messages)
    return normalize_action(parsed, state.tools), source


def heuristic_action(state: ConversationState, incoming_text: str) -> tuple[dict[str, Any], str]:
    if tool_result := extract_tool_result(incoming_text):
        tool_name_value, payload = tool_result
        return heuristic_tool_result_action(tool_name_value, payload), "heuristic:tool_result"
    user_text = extract_latest_user_text(incoming_text)
    if state.tools:
        if action := heuristic_tool_call(user_text, state.tools):
            return action, "heuristic:tool_call"
    return render_action(RESPOND_ACTION_NAME, {"content": "Please share the task details you want me to resolve."}), "heuristic:clarify"


def decide_next_action(state: ConversationState, incoming_text: str) -> tuple[dict[str, Any], dict[str, Any]]:
    tools = extract_tool_schemas(incoming_text)
    if tools:
        state.tools = tools
    if extract_tool_result(incoming_text):
        action, source = heuristic_action(state, incoming_text)
        return action, {"action_source": source}

    heuristic_candidate, heuristic_source = heuristic_action(state, incoming_text)
    if heuristic_candidate.get("name") != RESPOND_ACTION_NAME and os.environ.get("OPENCLAW_AGENTBEATS_FORCE_HEURISTIC", "").strip() == "1":
        return heuristic_candidate, {"action_source": heuristic_source}

    if os.environ.get("OPENCLAW_AGENTBEATS_DISABLE_MODEL", "").strip() != "1":
        try:
            action, source = model_action(state, incoming_text)
            if action.get("name") == RESPOND_ACTION_NAME and heuristic_candidate.get("name") != RESPOND_ACTION_NAME:
                return heuristic_candidate, {"action_source": heuristic_source}
            return action, {"action_source": source}
        except Exception:
            pass

    return heuristic_candidate, {"action_source": heuristic_source}


def action_from_task(task: dict[str, Any]) -> dict[str, Any]:
    for artifact in task.get("artifacts", []):
        if not isinstance(artifact, dict):
            continue
        for part in artifact.get("parts", []):
            if not isinstance(part, dict):
                continue
            text = str(part.get("text") or "").strip()
            if not text:
                continue
            try:
                parsed = json.loads(text)
            except Exception:
                continue
            if isinstance(parsed, dict) and "name" in parsed:
                return parsed
    return {}


class PurpleAgentServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, server_address: tuple[str, int], request_handler: type[BaseHTTPRequestHandler], *, card_url: str = "", repo_url: str = "") -> None:
        super().__init__(server_address, request_handler)
        self.card_url = normalize_base_url(card_url)
        self.repo_url = repo_url
        self.tasks: dict[str, dict[str, Any]] = {}
        self.conversations: dict[str, ConversationState] = {}

    def base_url_for(self, handler: BaseHTTPRequestHandler) -> str:
        if self.card_url:
            return self.card_url
        host = handler.headers.get("Host") or f"{self.server_address[0]}:{self.server_address[1]}"
        return f"http://{host}"

    def conversation(self, context_id: str) -> ConversationState:
        if context_id not in self.conversations:
            self.conversations[context_id] = ConversationState(context_id=context_id)
        return self.conversations[context_id]


class PurpleAgentHandler(BaseHTTPRequestHandler):
    server: PurpleAgentServer

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def send_json(self, payload: dict[str, Any], *, status: int = 200) -> None:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or "0")
        raw = self.rfile.read(length) if length else b"{}"
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except Exception:
            parsed = {}
        return parsed if isinstance(parsed, dict) else {}

    def do_GET(self) -> None:
        if self.path in {"/.well-known/agent-card.json", "/agent-card.json"}:
            card = build_agent_card(self.server.base_url_for(self), repo_url=self.server.repo_url)
            self.send_json(card)
            return
        if self.path.startswith("/tasks/"):
            task_id = self.path.split("/tasks/", 1)[1].split("?", 1)[0]
            task = self.server.tasks.get(task_id)
            if task is None:
                self.send_json({"error": "task_not_found", "task_id": task_id}, status=404)
                return
            self.send_json(task)
            return
        self.send_json({"error": "not_found", "path": self.path}, status=404)

    def do_POST(self) -> None:
        if self.path != "/message:send":
            self.send_json({"error": "not_found", "path": self.path}, status=404)
            return
        payload = self.read_json_body()
        task_id = str(uuid.uuid4())
        context_id = extract_context_id(payload) or str(uuid.uuid4())

        if looks_like_secret_probe(payload):
            task = build_refusal_task(task_id, context_id)
        elif looks_like_assessment_request(payload):
            task = build_task(task_id, extract_assessment_request(payload), context_id=context_id)
        else:
            incoming_text = extract_message_text(payload)
            state = self.server.conversation(context_id)
            action, metadata = decide_next_action(state, incoming_text)
            state.append("user", incoming_text)
            state.append("assistant", json.dumps(action, ensure_ascii=False))
            task = build_action_task(task_id, context_id, action, metadata)
        self.server.tasks[task_id] = task
        self.send_json({"task": task})


def start_server(host: str, port: int, *, card_url: str = "", repo_url: str = "") -> PurpleAgentServer:
    return PurpleAgentServer((host, port), PurpleAgentHandler, card_url=card_url, repo_url=repo_url)


def run_server(host: str, port: int, *, card_url: str = "", repo_url: str = "") -> None:
    server = start_server(host, port, card_url=card_url, repo_url=repo_url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local AgentBeats/OpenClaw purple agent with tau2 action-loop support.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve_parser = subparsers.add_parser("serve")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--card-url", default="")
    serve_parser.add_argument("--repo-url", default="")

    card_parser = subparsers.add_parser("card")
    card_parser.add_argument("--card-url", default="http://127.0.0.1:8000")
    card_parser.add_argument("--repo-url", default="")

    solve_parser = subparsers.add_parser("solve")
    solve_parser.add_argument("--text", required=True)
    solve_parser.add_argument("--context-id", default="cli")
    solve_parser.add_argument("--json", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "serve":
        run_server(args.host, args.port, card_url=args.card_url, repo_url=args.repo_url)
        return 0
    if args.command == "card":
        print(json.dumps(build_agent_card(args.card_url, repo_url=args.repo_url), ensure_ascii=False, indent=2))
        return 0
    if args.command == "solve":
        state = ConversationState(context_id=args.context_id)
        action, metadata = decide_next_action(state, args.text)
        payload = {
            "action": action,
            "metadata": metadata,
        }
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(payload, ensure_ascii=False))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
