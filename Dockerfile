# Stage 1: Builder to install dependencies
FROM python:3.9-slim AS builder

WORKDIR /app

# Install only needed build tools
RUN apt-get update && apt-get install -y build-essential

# Install torch from local wheel
COPY torch-1.13.1+cpu-cp39-cp39-linux_x86_64.whl .
RUN pip install --no-cache-dir torch-1.13.1+cpu-cp39-cp39-linux_x86_64.whl

# Copy only minimal requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Clean build tools
RUN apt-get purge -y build-essential && apt-get autoremove -y && apt-get clean

# Stage 2: Final minimal image
FROM python:3.9-slim

WORKDIR /app

# Copy only what's needed
COPY --from=builder /usr/local /usr/local

# Copy your actual code and local model
COPY process_documents.py .
COPY models/ ./models/

# Cleanup: remove heavy unused things
RUN rm -rf /usr/local/lib/python3.9/site-packages/*-tests \
           /usr/local/lib/python3.9/site-packages/__pycache__ \
           /usr/local/lib/python3.9/site-packages/nltk* \
           /usr/local/lib/python3.9/site-packages/sklearn* \
           /usr/local/lib/python3.9/site-packages/scipy* \
           /usr/local/lib/python3.9/site-packages/PIL*

CMD ["python", "process_documents.py"]
