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

wget -O $INSTALL_DIR/docker-compose.yaml https://raw.githubusercontent.com/Artrajz/vits-simple-api/main/docker-compose.yaml

echo -e "${YELLOW}Pulling the image might take a while, so why not grab a cup of java first?\n${PLAIN}"

docker compose pull
docker compose up -d

echo -e "\nThe upgrade or installation has been completed."
echo -e "The configuration file directory is $(realpath $INSTALL_DIR)"
echo -e "${YELLOW}If the vits model is not imported, it cannot be used. Import the model in the configuration file directory.${PLAIN}"
echo -e "After modifying the configuration file, restart the docker container for the modification to take effect."
echo -e "${YELLOW}If you have any questions, please put them in the issues.${PLAIN}"
echo -e "https://github.com/Artrajz/vits-simple-api"