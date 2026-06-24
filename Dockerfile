FROM python:3.12-slim AS builder
RUN apt-get update && apt-get install -y --no-install-recommends build-essential python3-dev
WORKDIR /build
COPY requirements.txt .
RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.12-slim
WORKDIR /app
COPY --from=builder /venv /venv
COPY app.py .
ENV PATH="/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
EXPOSE 8080
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--timeout", "120", "--capture-output", "--log-level", "info", "--access-logfile", "-", "--error-logfile", "-", "app:app"]
