FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-devel as base

RUN mkdir /work/
WORKDIR /work/

COPY ./requirements.txt /work/requirements.txt
RUN pip install -r requirements.txt

COPY ./src/ /work/

###########START NEW IMAGE : DEBUGGER ###################
FROM base as debug
RUN pip install ptvsd

WORKDIR /work/
CMD python -m ptvsd --host 0.0.0.0 --port 5678 --wait main.py

###########START NEW IMAGE: PRODUCTION ###################
FROM base as prod

CMD python -m main.py