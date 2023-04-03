FROM python:3.9.16-slim-bullseye

RUN mkdir -p /app
WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt install build-essential -yq && \
    apt-get clean && \
    apt-get purge -y --auto-remove -o APT::AutoRemove::RecommendsImportant=false && \
    rm -rf /var/lib/apt/lists/*

RUN pip install numpy==1.23.3 scipy flask flask_apscheduler pilk

RUN pip install numba librosa torch av

RUN pip install unidecode jamo pypinyin jieba protobuf cn2an inflect eng_to_ipa ko_pron indic_transliteration num_thai opencc

RUN pip install audonnx

COPY requirements.txt /app
RUN pip install -r requirements.txt

COPY . /app

EXPOSE 23456

CMD ["python", "/app/app.py"]

