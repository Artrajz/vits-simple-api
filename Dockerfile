FROM python:3.10.11-slim-bullseye

RUN mkdir -p /app
WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive

COPY . /app

RUN apt-get update && \
    apt-get install -yq build-essential espeak-ng cmake wget && \
    apt-get clean && \
    apt-get purge -y --auto-remove -o APT::AutoRemove::RecommendsImportant=false && \
    rm -rf /var/lib/apt/lists/*

RUN pip install torch --index-url https://download.pytorch.org/whl/cpu --no-cache-dir

RUN pip install pyopenjtalk==0.3.2 -i https://pypi.artrajz.cn/simple --no-cache-dir && \
    pip install -r requirements.txt --no-cache-dir -f https://pypi.artrajz.cn/simple && \
    pip install gunicorn --no-cache-dir

EXPOSE 23456

CMD ["gunicorn", "-c", "gunicorn_config.py", "app:app"]