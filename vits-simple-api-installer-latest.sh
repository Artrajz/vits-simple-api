INSTALL_DIR=/usr/local/vits-simple-api

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
PLAIN='\033[0m'

mkdir -p $INSTALL_DIR
cd $INSTALL_DIR

download_with_fallback() {
    local filename=$1
    shift  # Shift arguments to the left to handle URLs

    local success=0
    local url
    for url in "$@"; do
        echo -e "${YELLOW}Attempting to download $filename from $url\n${PLAIN}"
        if wget -O "$INSTALL_DIR/$filename" "$url"; then
            success=1
            break
        fi
    done

    if [ "$success" -ne 1 ]; then
        echo -e "${RED}Failed to download $filename from all provided URLs.${PLAIN}"
        exit 1
    fi
}

version_gt() {
    test "$(echo "$@" | tr " " "\n" | sort -V | head -n 1)" != "$1"
}

if [ ! -f config.py ]; then
    download_with_fallback config.py \
        "https://raw.githubusercontent.com/Artrajz/vits-simple-api/main/config.py" \
        "https://ghproxy.com/https://raw.githubusercontent.com/Artrajz/vits-simple-api/main/config.py"
fi

if [ ! -f gunicorn_config.py ]; then
    download_with_fallback gunicorn_config.py \
        "https://raw.githubusercontent.com/Artrajz/vits-simple-api/main/gunicorn_config.py" \
        "https://ghproxy.com/https://raw.githubusercontent.com/Artrajz/vits-simple-api/main/gunicorn_config.py"
fi

while true; do
    echo -e "${GREEN}Which version of docker-compose.yaml do you want to download?"
    echo -e "1. docker-compose.yaml (CPU version)"
    echo -e "2. docker-compose-gpu.yaml (GPU version)"
    read -p "Enter your choice (1 or 2): " choice
    case $choice in
        1)
            download_with_fallback docker-compose.yaml \
                "https://raw.githubusercontent.com/Artrajz/vits-simple-api/main/docker-compose.yaml" \
                "https://ghproxy.com/https://raw.githubusercontent.com/Artrajz/vits-simple-api/main/docker-compose.yaml"
            break
            ;;
        2)
            download_with_fallback docker-compose.yaml \
                "https://raw.githubusercontent.com/Artrajz/vits-simple-api/main/docker-compose-gpu.yaml" \
                "https://ghproxy.com/https://raw.githubusercontent.com/Artrajz/vits-simple-api/main/docker-compose-gpu.yaml"
            break
            ;;
        *)
            echo -e "${RED}Invalid choice. Please enter 1 or 2.${PLAIN}"
            ;;
    esac
done

if [ "$choice" -eq 2 ]; then
    DOCKER_VERSION=$(docker version --format '{{.Server.Version}}')
    MIN_DOCKER_VERSION="19.03"

    if version_gt $MIN_DOCKER_VERSION $DOCKER_VERSION; then
        echo -e "${RED}Your Docker version ($DOCKER_VERSION) does not support GPU. You need at least version $MIN_DOCKER_VERSION.${PLAIN}"
        exit 1
    fi
fi

if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}docker-compose could not be found.${PLAIN}"
    exit 1
fi

echo -e "${YELLOW}Pulling the image might take a while, so why not grab a cup of java first?\n${PLAIN}"

docker compose pull
docker compose up -d

if [ "$choice" -eq 2 ]; then
    if ! docker run --gpus all artrajz/vits-simple-api:latest-gpu nvidia-smi &> /dev/null; then
        echo -e "${RED}Your Docker does not seem to support GPU or NVIDIA Docker is not installed properly.${PLAIN}"
        exit 1
    fi
fi

echo -e "\nThe upgrade or installation has been completed."
echo -e "The configuration file directory is $(realpath $INSTALL_DIR)"
echo -e "${YELLOW}If the vits model is not imported, it cannot be used. Import the model in the configuration file directory.${PLAIN}"
echo -e "After modifying the configuration file, restart the docker container for the modification to take effect."
echo -e "${YELLOW}If you have any questions, please put them in the issues.${PLAIN}"
echo -e "https://github.com/Artrajz/vits-simple-api"