# Use an official Python runtime as a parent image
FROM python:3.11-slim-buster

# Set the working directory in the container to /backend-dais
WORKDIR /dais

# Copy the current directory contents into the container at /backend-dais
COPY . /dais

# Install any needed packages specified in requirements.txt
RUN apt-get update && \
    apt-get install -y git && \
    pip install --no-cache-dir -r requirements.txt

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Run main.py when the container launches
CMD ["python", "./main.py"]

# docker build -t backend-dais:latest .
# don't forget the full stop above for it to actually build it.
# check model locations before creating. "models/model_name" no ../ needed. Should be needed with .env