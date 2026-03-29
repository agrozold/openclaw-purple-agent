# OpenClaw Purple Agent

This repository is the public AgentX-AgentBeats submission package for the OpenClaw purple agent.

## What It Does

OpenClaw focuses on bounded operator workflows:

- bounty triage
- repository execution planning
- browser-assisted research
- truthful readiness and blocker reporting

The public package exposes a minimal A2A-compatible HTTP surface for local smoke tests and competition submissions.

## Files

- `agentbeats_purple_agent.py`: minimal A2A-compatible purple agent
- `Dockerfile`: container entrypoint for assessment runs
- `openclaw-purple-agent.json5`: Amber component manifest for AgentBeats registration
- `abstract.md`: short submission abstract
- `scenario.template.toml`: starting point for manual or local leaderboard runs

## Local Run

```bash
python3 agentbeats_purple_agent.py serve --host 127.0.0.1 --port 8000
```

## Local Card

```bash
python3 agentbeats_purple_agent.py card --card-url http://127.0.0.1:8000
```

## Docker Build

```bash
docker build -t openclaw-purple-agent:local .
```

## Notes

- This public repo is intentionally minimal and excludes private workspace state.
- Submission still requires a public GitHub remote, a public image path, and registration on `agentbeats.dev`.
