FROM artrajz/pytorch:1.13.1-cpu-py3.10.11-ubuntu22.04

RUN mkdir -p /app
WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive


RUN apt-get update && \
    apt-get install -yq build-essential espeak-ng cmake wget ca-certificates tzdata&& \
    update-ca-certificates && \
    apt-get clean && \
    apt-get purge -y --auto-remove -o APT::AutoRemove::RecommendsImportant=false && \
    rm -rf /var/lib/apt/lists/* 

# Install jemalloc
RUN wget https://github.com/jemalloc/jemalloc/releases/download/5.3.0/jemalloc-5.3.0.tar.bz2 && \
    tar -xvf jemalloc-5.3.0.tar.bz2 && \
    cd jemalloc-5.3.0 && \
    ./configure && \
    make -j$(nproc) && \
    make install && \
    cd .. && \
    rm -rf jemalloc-5.3.0* && \
    ldconfig

ENV LD_PRELOAD=/usr/local/lib/libjemalloc.so

COPY requirements.txt /app/
RUN pip install gunicorn --no-cache-dir && \
    pip install -r requirements.txt --no-cache-dir&& \
    rm -rf /root/.cache/pip/*

COPY . /app

EXPOSE 23456

CMD ["gunicorn", "-c", "gunicorn_config.py", "app:app"]