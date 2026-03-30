#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import uuid
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


def build_agent_card(base_url: str, repo_url: str = "") -> dict[str, Any]:
    service_url = base_url.rstrip("/") + "/message:send"
    return {
        "name": "OpenClaw Purple Agent",
        "description": "Telegram-first operator agent for bounded bounty triage, execution planning, and truthful readiness reporting.",
        "url": service_url,
        "provider": {
            "organization": "OpenClaw",
            "url": repo_url or "",
        },
        "version": "0.1.0",
        "capabilities": {
            "streaming": False,
            "pushNotifications": False,
        },
        "defaultInputModes": ["text/plain", "application/json"],
        "defaultOutputModes": ["application/json", "text/plain"],
        "skills": [
            {
                "id": "bounty-triage",
                "name": "Bounty triage",
                "description": "Evaluates scoped GitHub issues and prepares safe execution plans.",
            },
            {
                "id": "operator-status",
                "name": "Operator status",
                "description": "Reports blockers, manual gates, and readiness honestly.",
            },
        ],
    }


def parse_text_parts(message: dict[str, Any]) -> list[str]:
    parts = message.get("parts", []) if isinstance(message, dict) else []
    result: list[str] = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        text = str(part.get("text") or "").strip()
        if text:
            result.append(text)
    return result


def looks_like_secret_probe(payload: dict[str, Any]) -> bool:
    fragments: list[str] = []
    message = payload.get("message", {})
    if isinstance(message, dict):
        fragments.extend(parse_text_parts(message))
    metadata = payload.get("metadata", {})
    if isinstance(metadata, dict):
        for key in ("prompt", "instruction", "request", "assessment_request"):
            value = metadata.get(key)
            if isinstance(value, dict):
                fragments.append(json.dumps(value, ensure_ascii=False))
            elif isinstance(value, str):
                fragments.append(value)
    blob = " ".join(part.lower() for part in fragments if part).strip()
    if not blob:
        return False
    return any(pattern in blob for pattern in SENSITIVE_PROBE_PATTERNS)


def extract_assessment_request(payload: dict[str, Any]) -> dict[str, Any]:
    if isinstance(payload.get("metadata"), dict):
        metadata = payload["metadata"]
        assessment = metadata.get("assessment_request")
        if isinstance(assessment, dict):
            return assessment
    message = payload.get("message", {})
    for text in parse_text_parts(message if isinstance(message, dict) else {}):
        try:
            parsed = json.loads(text)
        except Exception:
            continue
        if isinstance(parsed, dict) and "participants" in parsed:
            return parsed
    return {
        "participants": {},
        "config": {},
    }


def build_refusal_task(task_id: str) -> dict[str, Any]:
    report = {
        "agent": "openclaw-purple",
        "summary": "OpenClaw refused a request for secrets or local filesystem details.",
        "policy": [
            "Do not disclose wallets, mnemonics, private keys, SSH material, or secrets.env contents.",
            "Do not reveal arbitrary local filesystem paths or file contents.",
            "Provide only bounded public status and competition metadata.",
        ],
        "generated_at": utc_now_iso(),
    }
    return {
        "id": task_id,
        "kind": "task",
        "status": {
            "state": "completed",
            "timestamp": utc_now_iso(),
        },
        "artifacts": [
            {
                "name": "openclaw-safety-refusal",
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
            "policy_decision": "refused_sensitive_request",
        },
    }


def build_task(task_id: str, assessment_request: dict[str, Any]) -> dict[str, Any]:
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


class PurpleAgentServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, server_address: tuple[str, int], request_handler: type[BaseHTTPRequestHandler], *, card_url: str = "", repo_url: str = "") -> None:
        super().__init__(server_address, request_handler)
        self.card_url = normalize_base_url(card_url)
        self.repo_url = repo_url
        self.tasks: dict[str, dict[str, Any]] = {}

    def base_url_for(self, handler: BaseHTTPRequestHandler) -> str:
        if self.card_url:
            return self.card_url
        host = handler.headers.get("Host") or f"{self.server_address[0]}:{self.server_address[1]}"
        return f"http://{host}"


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
        if looks_like_secret_probe(payload):
            task = build_refusal_task(task_id)
        else:
            assessment_request = extract_assessment_request(payload)
            task = build_task(task_id, assessment_request)
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
    parser = argparse.ArgumentParser(description="Minimal local AgentBeats/OpenClaw purple-agent surface.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve_parser = subparsers.add_parser("serve")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--card-url", default="")
    serve_parser.add_argument("--repo-url", default="")

    card_parser = subparsers.add_parser("card")
    card_parser.add_argument("--card-url", default="http://127.0.0.1:8000")
    card_parser.add_argument("--repo-url", default="")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "serve":
        run_server(args.host, args.port, card_url=args.card_url, repo_url=args.repo_url)
        return 0
    if args.command == "card":
        print(json.dumps(build_agent_card(args.card_url, repo_url=args.repo_url), ensure_ascii=False, indent=2))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
