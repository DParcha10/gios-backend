# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Install system dependencies for geospatial libraries
RUN apt-get update && apt-get install -y \
    binutils \
    libproj-dev \
    gdal-bin \
    libgdal-dev \
    python3-gdal \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for GDAL
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

WORKDIR /app

# Upgrade pip and install build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install dependencies one by one to avoid version conflicts
COPY requirements.txt .
RUN pip install --no-cache-dir numpy==1.26.4
RUN pip install --no-cache-dir rasterio==1.3.10
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Use the PORT environment variable provided by Render
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
