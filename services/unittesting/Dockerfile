# Use an official Python runtime as a parent image
FROM python:3.10.13-bookworm

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip
RUN pip install pytest coverage

# Run linting with flake8

RUN pip install flake8

# Set the working directory to /app/MapAction
# Install coverage and run tests
RUN pip install coverage
RUN coverage run --omit=*/python3/* -m pytest
RUN coverage report
