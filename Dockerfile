FROM python:3.10.11-slim-bullseye

RUN mkdir -p /app
WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt install build-essential -yq && \
    apt install espeak-ng -yq && \
    apt-get clean && \
    apt-get purge -y --auto-remove -o APT::AutoRemove::RecommendsImportant=false && \
    rm -rf /var/lib/apt/lists/*

RUN pip install torch

COPY requirements.txt /app
RUN pip install -r requirements.txt

COPY . /app

EXPOSE 23456

CMD ["python", "/app/app.py"]

