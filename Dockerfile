# Use python 3.11 as base image
FROM python:3.11.4-slim-bullseye

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Download and cache dependencies
RUN pip install -r requirements.txt

CMD [ "python", "/app/src/api.py" ]
