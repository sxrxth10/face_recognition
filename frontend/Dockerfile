FROM python:3.9

WORKDIR /app

# Copy and install dependencies first for better caching
COPY frontend/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy frontend code
COPY frontend /app

EXPOSE 8501

CMD ["streamlit", "run", "frontend.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
