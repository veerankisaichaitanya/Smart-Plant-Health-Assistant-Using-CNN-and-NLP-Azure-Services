# Use Python 3.9 as the base image
FROM python:3.9-slim

# Set the working directory inside the container to /app
WORKDIR /app

# Copy the requirements.txt file to the container
COPY requirements.txt .

# Install the Python libraries listed in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy all the project files to the container (including your FastAPI code)
COPY . .

# Expose port 8000 (the port your FastAPI app will run on)
EXPOSE 8000

# Run the FastAPI app with uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
