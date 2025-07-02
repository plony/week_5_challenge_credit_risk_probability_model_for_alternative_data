# Dockerfile

# Use an official Python runtime as a parent image
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app

# Install system dependencies needed for some Python packages (e.g., psycopg2-binary for sqlite)
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    # Add any other system dependencies if your specific ML libraries require them
    # For example, if you were using xgboost you might need libgomp1
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the working directory
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project directory into the container
# This includes src/, tests/, and any other necessary files like processed_data_with_proxy_target.csv
# Make sure to copy processed_data_with_proxy_target.csv and mlruns/ for the model loading
COPY . /app

# Expose the port that Uvicorn will run on
EXPOSE 8000

# Command to run the application using Uvicorn
# Make sure main.py is in src/api/main.py, so it's src.api.main
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]