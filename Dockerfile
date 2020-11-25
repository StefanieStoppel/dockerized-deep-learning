# Use a base image which contains PyTorch 1.7.0, CUDA 11.0 and cuDNN 8 (CUDA Deep Neural Network library)
FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-devel

# Specify where our MNIST data set should be downloaded to
ENV DATA_PATH="/data"

# Create /work and /data directories
RUN mkdir -p /work/ ${DATA_PATH}
WORKDIR /work/

# Install the Python debugger debugpy by Microsoft
RUN pip install debugpy

# Download the MNIST data set to ${DATA_PATH} using our custom Python function.
# If we don't do this here, the data set will be re-downloaded every time we run the container,
# since the container's file storage isn't persisted after it terminates.
# Read more about persisting files here: https://docs.docker.com/storage/
RUN python -c "from dataloaders import get_mnist_data_sets; get_mnist_data_sets('${DATA_PATH}')"

# Python debug command that is run when starting the container
CMD python -m debugpy --listen 0.0.0.0:5678 --wait-for-client main.py
