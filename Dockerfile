# ---- Builder stage ----
FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY requirements.txt .

RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip setuptools "setuptools<82" wheel
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision
RUN pip install --no-cache-dir -r requirements.txt

RUN find /venv -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true


# ---- Final stage ----
FROM python:3.12-slim

# All required runtime libraries for opencv-python-headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    fonts-dejavu-core \
    libglib2.0-0 \
    libxcb1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /venv /venv
COPY app.py .

ENV PATH="/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# Preload models
RUN python -c "import pyiqa; pyiqa.create_metric('nima');"
RUN python -c "import pyiqa; pyiqa.create_metric('brisque');"

EXPOSE 8080
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--timeout", "300", "--capture-output", "--log-level", "info", "--access-logfile", "-", "--error-logfile", "-", "app:app"]
