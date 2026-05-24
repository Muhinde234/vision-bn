# All packages (psycopg2-binary, Pillow, opencv-headless, etc.) ship as
# pre-built binary wheels on PyPI — no gcc or system libs needed.
FROM python:3.11-slim-bookworm

WORKDIR /app

# Non-root user
RUN useradd --create-home appuser
USER appuser
ENV PATH="/home/appuser/.local/bin:$PATH"

# Install Python deps straight from PyPI (all binary wheels)
COPY requirements.prod.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.prod.txt

# Copy application code
COPY --chown=appuser:appuser . .

RUN mkdir -p /app/uploads /app/models

EXPOSE 8000

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

CMD uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers 4
