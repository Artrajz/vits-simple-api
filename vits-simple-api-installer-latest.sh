INSTALL_DIR=/usr/local/vits-simple-api

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
PLAIN='\033[0m'

mkdir -p $INSTALL_DIR
cd $INSTALL_DIR
if [ ! -f config.py ]; then
    echo -e "${YELLOW}download config.py\n${PLAIN}"
    wget -O $INSTALL_DIR/config.py https://raw.githubusercontent.com/Artrajz/vits-simple-api/main/config.py
fi

if [ ! -f gunicorn_config.py ]; then
    echo -e "${YELLOW}download config.py\n${PLAIN}"
    wget -O $INSTALL_DIR/gunicorn_config.py https://raw.githubusercontent.com/Artrajz/vits-simple-api/main/gunicorn_config.py
fi

while true; do
    echo -e "${GREEN}Which version of docker-compose.yaml do you want to download?"
    echo -e "1. docker-compose.yaml (CPU version)"
    echo -e "2. docker-compose-gpu.yaml (GPU version)"
    read -p "Enter your choice (1 or 2): " choice
    case $choice in
        1)
            echo -e "${YELLOW}Downloading docker-compose.yaml (CPU version)\n${PLAIN}"
            wget -O $INSTALL_DIR/docker-compose.yaml https://raw.githubusercontent.com/Artrajz/vits-simple-api/main/docker-compose.yaml
            break
            ;;
        2)
            echo -e "${YELLOW}Downloading docker-compose-gpu.yaml (GPU version)\n${PLAIN}"
            wget -O $INSTALL_DIR/docker-compose.yaml https://raw.githubusercontent.com/Artrajz/vits-simple-api/main/docker-compose-gpu.yaml
            break
            ;;
        *)
            echo -e "${RED}Invalid choice. Please enter 1 or 2.${PLAIN}"
            ;;
    esac
done

echo -e "${YELLOW}Pulling the image might take a while, so why not grab a cup of java first?\n${PLAIN}"

docker compose pull
docker compose up -d

echo -e "\nThe upgrade or installation has been completed."
echo -e "The configuration file directory is $(realpath $INSTALL_DIR)"
echo -e "${YELLOW}If the vits model is not imported, it cannot be used. Import the model in the configuration file directory.${PLAIN}"
echo -e "After modifying the configuration file, restart the docker container for the modification to take effect."
echo -e "${YELLOW}If you have any questions, please put them in the issues.${PLAIN}"
echo -e "https://github.com/Artrajz/vits-simple-api"