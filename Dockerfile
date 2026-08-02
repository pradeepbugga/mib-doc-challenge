FROM python:3.12-slim

# tesseract-ocr: OCR engine for scanned/degraded pages.
# tesseract-ocr-eng: English trained data (packets are all English).
# tesseract-ocr-osd: orientation/script detection, used for rotation retry.
# libgl1: opencv-python-headless still links against libGL at import time.
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-eng \
        tesseract-ocr-osd \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY core /app/core
COPY scripts /app/scripts
COPY run.sh /app/run.sh
RUN chmod +x /app/run.sh

# The scoring container mounts a read-only root filesystem; nothing under
# /app is written to at runtime. Predictions go to the output mount, and any
# scratch files the OCR/PDF libraries need go under /tmp (the writable tmpfs).
ENV TMPDIR=/tmp \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

ENTRYPOINT ["/app/run.sh"]
