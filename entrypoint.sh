#!/bin/sh

if [ ! -d "/app/data" ] || [ -z "$(ls -A /app/data)" ]; then
    echo "The host's ./data directory is empty or does not exist. Copying data from the container..."
    mkdir -p /app/data
    cp -r /data_bak/* /app/data/
fi

exec "$@"
