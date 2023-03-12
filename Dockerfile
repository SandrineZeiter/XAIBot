# Use Python 3.10.10 base image
FROM python:3.10.2-slim-buster

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any necessary dependencies
RUN apt-get update \
    && apt-get install -y libatlas-base-dev \
    && pip install --no-cache-dir -r requirements.txt

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable
ENV FLASK_APP=main.py

# Run the command to start the Flask application
CMD ["flask", "run", "--host", "0.0.0.0"]