# Use Python 3.12 slim image as a base
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any necessary dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Set environment variables to allow Flask to run properly
ENV FLASK_APP=app.py
ENV FLASK_ENV=production 

# Expose the port that Flask will run on
EXPOSE 5000

# Command to run the Flask application
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]
