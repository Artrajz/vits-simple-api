FROM python:3.10.11-slim-bullseye

RUN mkdir -p /app
WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt install build-essential -yq && \
    apt install espeak-ng -yq && \
    apt install cmake -yq && \
    apt install -y wget -yq && \
    apt-get clean && \
    apt-get purge -y --auto-remove -o APT::AutoRemove::RecommendsImportant=false && \
    rm -rf /var/lib/apt/lists/*

RUN pip install MarkupSafe==2.1.2 numpy==1.23.3 cython six==1.16.0

RUN wget https://raw.githubusercontent.com/Artrajz/archived/main/openjtalk/openjtalk-0.3.0.dev2.tar.gz && \
    tar -zxvf openjtalk-0.3.0.dev2.tar.gz && \
    cd openjtalk-0.3.0.dev2 && \
    rm -rf ./pyopenjtalk/open_jtalk_dic_utf_8-1.11 && \
    python setup.py install && \
    cd ../ && \
    rm -f openjtalk-0.3.0.dev2.tar.gz && \
    rm -rf openjtalk-0.3.0.dev2

RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt /app
RUN pip install -r requirements.txt

RUN pip install gunicorn

COPY . /app

EXPOSE 23456

CMD ["gunicorn", "-c", "gunicorn_config.py", "app:app"]