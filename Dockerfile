FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose port (Streamlit default)
EXPOSE 8501

# Run the Streamlit application. Railway will inject a dynamic PORT env var, 
# so we fall back to 8501 if it's not present.
CMD ["sh", "-c", "streamlit run src/ui/app.py --server.port ${PORT:-8501} --server.address 0.0.0.0"]
