# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory to /app
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# INSTALL VIM
RUN apt-get update && apt-get install -y vim

# Copy aliases
COPY .bash_aliases /root/
RUN echo "if [ -f /root/.bash_aliases ]; then . /root/.bash_aliases; fi" >> /root/.bashrc

# Copy the rest of the application code into the container
COPY . .

# Expose port 5001 (the port that Flask uses by default)
EXPOSE 8080

# Set the command to run when the container starts
CMD ["gunicorn", "-b", ":8080", "main:app"]

# Set the command to run when the container starts
# CMD ["python", "main.py"]