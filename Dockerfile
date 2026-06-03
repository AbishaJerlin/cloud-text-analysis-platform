# Use official Python base image
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements first (so Docker can cache the install layer)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download spacy model
RUN python -m spacy download en_core_web_sm

# Copy everything else into the container
COPY . .

# Expose the port Gradio runs on
EXPOSE 7860

# Run the notebook as a script, or swap this for your entry point
CMD ["python", "app.py"]
