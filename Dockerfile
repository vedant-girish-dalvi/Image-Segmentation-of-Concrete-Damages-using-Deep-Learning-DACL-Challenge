# Use NVIDIA's official PyTorch + CUDA image (works with Enroot)
FROM nvcr.io/nvidia/pytorch:24.03-py3

# Metadata
LABEL maintainer="Vedant Dalvi>"

# Set working directory
WORKDIR /app

# The base image already includes CUDA, cuDNN, PyTorch, and Python.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Ensure environment is accessible in non-interactive shells
SHELL ["/bin/bash", "-lc"]

# Copy project files
COPY . /app

# Make all directories writable by non-root users (important for Enroot)
RUN chmod -R a+rwx /app

# Environment setup
ENV PYTHONUNBUFFERED=1 \
PATH=/opt/conda/envs/dacl_env/bin:$PATH

# Default command (interactive-friendly)
CMD ["bash"]