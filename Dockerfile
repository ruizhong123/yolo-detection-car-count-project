# Build stage
FROM python:3.12-slim AS builder

RUN mkdir /app
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install build dependencies
RUN apt-get update && apt-get install -y \
    python3-dev \
    libxml2-dev \
    libexpat1-dev \
    krb5-user \
    && apt-get install -y --only-upgrade \
    libxml2 \
    libexpat1 \
    libgssapi-krb5-2 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy and install requirements

COPY project/requirements.txt .
RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt

# Final stage
FROM python:3.12-slim

RUN mkdir /app
WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    libxml2 \
    libexpat1 \
    libgssapi-krb5-2 \
    && apt-get install -y --only-upgrade \
    libxml2 \
    libexpat1 \
    libgssapi-krb5-2 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m appuser && chown -R appuser /app

# Copy dependencies from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code and model
COPY --chown=appuser:appuser --chmod=755 project/ .
COPY --chown=appuser:appuser model/yolo12n.pt .

# Switch to non-root user
USER appuser

EXPOSE 8000

# run docker run -it project_version5-django_web sh
CMD ["/app/entrypoint.prod.sh"]


