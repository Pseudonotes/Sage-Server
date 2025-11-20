FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# System deps required by some libraries (adjust as needed)
# RUN apt-get update && \
#     apt-get install -y build-essential git ffmpeg libsndfile1 && \
#     rm -rf /var/lib/apt/lists/*

# Copy and install python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# Copy app code
COPY . /app

# Entrypoint will wait for dependent services before starting the app
ENTRYPOINT ["python", "/app/entrypoint.py"]
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]