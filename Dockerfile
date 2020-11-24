########### Base image ###########
FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-devel as base

ENV DATA_PATH="/data"

RUN mkdir -p /work/ ${DATA_PATH}
WORKDIR /work/

COPY ./src/ /work/

# Download the MNIST data set to ${DATA_PATH} using our custom Python function.
# If we don't do this here, the data set will be re-downloaded every time we run the container,
# since the container's file storage isn't persisted after it's removed.
# Read more about this here: https://docs.docker.com/storage/
RUN python -c "from dataloaders import get_mnist_data_sets; get_mnist_data_sets('${DATA_PATH}')"

########### START NEW IMAGE : DEBUGGER ###################
FROM base as debug
RUN pip install ptvsd

WORKDIR /work/
CMD python -m ptvsd --host 0.0.0.0 --port 5678 --wait main.py

########### START NEW IMAGE: PRODUCTION ###################
FROM base as prod

CMD python -m main.py