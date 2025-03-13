# Use an official Python base image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy only the requirements file first (for caching benefits)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . .

# Explicitly copy the model folder
COPY model /app/model


# Expose Flask's default port
EXPOSE 5000

# Run Flask app
CMD ["python", "app/app.py"]
