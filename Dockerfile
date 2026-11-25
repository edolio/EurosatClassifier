# EuroSAT land-cover web app with both shipped models. No dataset inside: classify uploaded tiles;
# mount the dataset at /app/data/EuroSAT_MS to train or label (see README, "Run with Docker").
FROM python:3.12-slim

# uv from its official image, pinned: installs the exact versions in uv.lock
COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/opt/venv PATH="/opt/venv/bin:$PATH"

# Dependencies first, so code changes don't reinstall them
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --no-install-project --no-dev

COPY . .

# The app reads models/ and static/ next to app.py; listen on all interfaces so -p 5000:5000 reaches it
ENV HOST=0.0.0.0 PORT=5000 MPLBACKEND=Agg
EXPOSE 5000
CMD ["python", "app.py"]
