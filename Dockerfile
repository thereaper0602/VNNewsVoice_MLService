FROM python:3.12-slim

WORKDIR /app

# Cài đặt các dependencies hệ thống cần thiết
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Cài đặt dependencies cơ bản (nhẹ) trong build time
COPY requirements-base.txt .
RUN pip install --no-cache-dir -r requirements-base.txt

# Copy toàn bộ code
COPY . .

# Biến môi trường 
ENV PYTHONUNBUFFERED=1
ENV PORT=8000
ENV TOKENIZERS_PARALLELISM=false
ENV OMP_NUM_THREADS=1

# Expose port
EXPOSE 8000

# Cài đặt ML dependencies trong runtime và chạy app
# Giải quyết vấn đề giới hạn kích thước image 4GB của Railway
CMD pip install --no-cache-dir -r requirements-ml.txt && \
    uvicorn app.main:app --host 0.0.0.0 --port 8000