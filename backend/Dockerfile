FROM python:3.9

WORKDIR /app

# Copy and install dependencies first for better caching
COPY backend/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code and models
COPY backend /app
COPY models /app/models 

EXPOSE 8000

CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
