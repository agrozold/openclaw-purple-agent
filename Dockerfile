FROM python:3.10-slim

WORKDIR /app

COPY agentbeats_purple_agent.py /app/agentbeats_purple_agent.py

ENTRYPOINT ["python", "/app/agentbeats_purple_agent.py", "serve"]
