# Base Image: Lightweight Python 3.11
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Copy dependency file first (for caching)
COPY requirements.txt .

# Install dependencies
# We upgrade pip to avoid warnings and install requirements
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port Chainlit runs on
EXPOSE 8000

# Command to run the app
# -h 0.0.0.0 is crucial for Docker to listen on external interfaces
CMD ["chainlit", "run", "app.py", "-h", "0.0.0.0", "-p", "8000"]