FROM python:3.11.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app

EXPOSE 7860

# FIXED: Point to env.py where your FastAPI app is
CMD ["uvicorn", "env:app", "--host", "0.0.0.0", "--port", "7860"]
