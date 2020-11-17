FROM python:3.8.6-alpine3.12 as base

RUN mkdir /work/
WORKDIR /work/

COPY ./requirements.txt /work/requirements.txt
RUN pip install -r requirements.txt

COPY ./src/ /work/
ENV FLASK_APP=server.py

###########START NEW IMAGE : DEBUGGER ###################
FROM base as debug
RUN pip install ptvsd

WORKDIR /work/
CMD python -m ptvsd --host 0.0.0.0 --port 5678 --wait solver.py

###########START NEW IMAGE: PRODUCTION ###################
FROM base as prod

CMD python -m solver.py