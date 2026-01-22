# Use Python 3.11 base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose configurable port (default 8000)
EXPOSE 8000

# Set entrypoint to run uvicorn server
CMD ["python", "main.py"]
