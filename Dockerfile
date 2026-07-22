FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

EXPOSE 8000

CMD ["sh", "-c", "echo 'Booting uvicorn on port' $PORT && exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --log-level info"]
