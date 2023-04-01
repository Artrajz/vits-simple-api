FROM python:3.9.16-slim-bullseye

RUN mkdir -p /app
WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install -yq build-essential
COPY requirements.txt /app
RUN pip install -r requirements.txt

COPY . /app

EXPOSE 23456

CMD ["python", "/app/app.py"]

