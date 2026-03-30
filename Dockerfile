FROM python:3.10-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY agentbeats_purple_agent.py /app/agentbeats_purple_agent.py

ENTRYPOINT ["python", "/app/agentbeats_purple_agent.py", "serve"]
