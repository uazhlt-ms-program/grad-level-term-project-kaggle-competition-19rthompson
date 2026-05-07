FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (better caching)
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy project into container
COPY . .

# Default command (you can change this if needed)
CMD ["python3", "test_models.py"]