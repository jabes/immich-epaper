FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
ENV PYTHONUNBUFFERED 1
COPY app.py .
EXPOSE 8080
CMD ["gunicorn", "--bind", "0.0.0.0:8081", "--workers", "1", "--timeout", "120", "--capture-output", "--log-level", "debug", "--access-logfile", "-", "--error-logfile", "-", "app:app"]
