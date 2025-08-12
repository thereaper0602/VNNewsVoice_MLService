FROM python:3.12-slim

# Cài đặt các gói cần thiết
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Cài đặt dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ code
COPY . .

# Tối ưu cho PyTorch và transformers
ENV PYTHONUNBUFFERED=1
ENV TOKENIZERS_PARALLELISM=false
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

# Expose port
ENV PORT=8000

# Lệnh khởi động
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]