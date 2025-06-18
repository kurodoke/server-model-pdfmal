FROM python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System dependency minimal (cukup gcc untuk joblib/sklearn)
RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

CMD ["which", "gunicorn"]

COPY . .

ENV PORT=5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers=1", "--threads=4", "--timeout=0", "app:app"]
