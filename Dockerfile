FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY tests ./tests
COPY run.py ./
ENV PYTHONPATH=/app/src:/app
ENV FLASHBULLJEV_BACKEND=fake
ENV OLLAMA_URL=http://host.docker.internal:11434
EXPOSE 8018
CMD ["uvicorn", "flashbulljev.api:app", "--host", "0.0.0.0", "--port", "8018"]
