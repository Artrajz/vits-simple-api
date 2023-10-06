INSTALL_DIR=/usr/local/vits-simple-api

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
PLAIN='\033[0m'

declare -A EN_MESSAGES
declare -A ZH_MESSAGES

EN_MESSAGES=(
  ["ATTEMPT_DOWNLOAD"]="Attempting to download"
  ["FROM"]="from"
  ["DOWNLOAD_FAIL"]="Failed to download"
  ["FROM_ALL_URLS"]="from all provided URLs."
  ["DOWNLOADING"]="Downloading..."
  ["VERIFYING"]="Verifying..."
  ["UNZIPPING"]="Unzipping..."
  ["CHOOSE_VERSION"]="Which version of docker-compose.yaml do you want to download?"
  ["DOCKER_CPU"]="docker-compose.yaml (CPU version)"
  ["DOCKER_GPU"]="docker-compose-gpu.yaml (GPU version)"
  ["ENTER_CHOICE"]="Enter your choice (1 or 2): "
  ["INVALID_CHOICE"]="Invalid choice. Please enter 1 or 2."
  ["DOWNLOAD_CONFIG"]="Downloading configuration file shortly..."
  ["PULL_IMAGE"]="Do you want to start pulling the image? Enter 1 for yes or 2 for no"
  ["DOWNLOAD_DICT"]="Do you want to download the pyopenjtalk dictionary file? Enter 1 for yes or 2 for no"
  ["MUST_DOWNLOAD_JP"]="Japanese model must be downloaded."
  ["DOWNLOAD_VITS_CHINESE"]="Do you want to download the bert model for vits_chinese? Enter 1 for yes, 2 for no."
  ["MUST_DOWNLOAD_VITS_CHINESE"]="Using vits_chinese requires downloading these models, which will take up about 410MB."
  ["DOWNLOAD_BERT_VITS2"]="Do you want to download chinese-roberta-wwm-ext-large? Enter 1 for yes or 2 for no"
  ["MUST_DOWNLOAD_BERT_VITS2"]="To use Bert-VITS2, you must download these models, which will take up about 1.63GB."
  ["DOWNLOADED"]="File is downloaded correctly."
  ["CORRUPTED"]="The file may not have been downloaded, or the download might be incomplete, and it could also be corrupted."
  ["INSTALL_COMPLETE"]="The upgrade or installation has been completed."
  ["CONFIG_DIR"]="The configuration file directory is"
  ["IMPORT_NOTICE"]="If the vits model is not imported, it cannot be used. Import the model in the configuration file directory."
  ["RESTART_NOTICE"]="After modifying the configuration file, restart the docker container for the modification to take effect."
  ["ISSUE_NOTICE"]="If you have any questions, please put them in the issues."
  ["GITHUB_LINK"]="https://github.com/Artrajz/vits-simple-api"
)

ZH_MESSAGES=(
  ["ATTEMPT_DOWNLOAD"]="正在尝试下载"
  ["FROM"]="从"
  ["DOWNLOAD_FAIL"]="都下载失败"
  ["FROM_ALL_URLS"]="从所有提供的URLs"
  ["DOWNLOADING"]="正在下载..."
  ["VERIFYING"]="正在校验"
  ["UNZIPPING"]="正在解压..."
  ["CHOOSE_VERSION"]="你想下载哪个版本的docker-compose.yaml？"
  ["DOCKER_CPU"]="docker-compose.yaml (CPU版本)"
  ["DOCKER_GPU"]="docker-compose-gpu.yaml (GPU版本)"
  ["ENTER_CHOICE"]="请输入您的选择 (1 或 2): "
  ["INVALID_CHOICE"]="无效选择。 请重新输入 1 或 2。"
  ["DOWNLOAD_CONFIG"]="即将下载配置文件..."
  ["PULL_IMAGE"]="是否要开始拉取镜像？输入1表示是，2表示否。"
  ["DOWNLOAD_DICT"]="是否要下载pyopenjtalk的词典文件？输入1表示是，2表示否。"
  ["MUST_DOWNLOAD_JP"]="使用日语模型必须下载该词典文件，将占用大约102MB。"
  ["DOWNLOAD_VITS_CHINESE"]="是否要下载vits_chinese的bert模型？输入1表示是，2表示否。"
  ["MUST_DOWNLOAD_VITS_CHINESE"]="使用vits_chinese必须下载这些模型，将占用大约410MB。"
  ["DOWNLOAD_BERT_VITS2"]="是否要下载chinese-roberta-wwm-ext-large？输入1表示是，2表示否。"
  ["MUST_DOWNLOAD_BERT_VITS2"]="使用Bert-VITS2必须下载这些模型，将占用大约1.63GB。"
  ["DOWNLOADED"]="文件已正确下载。"
  ["CORRUPTED"]="文件可能未下载，或下载不完整，也有可能已损坏。"
  ["INSTALL_COMPLETE"]="更新或安装已完成。"
  ["CONFIG_DIR"]="配置文件目录是"
  ["IMPORT_NOTICE"]="如果vits模型没有被导入，它是无法使用的。请在配置文件目录中导入模型。"
  ["RESTART_NOTICE"]="修改配置文件后，请重启docker容器以使修改生效。"
  ["ISSUE_NOTICE"]="如果你有任何问题，请在issues中提出，或者加入q群提问。"
  ["GITHUB_LINK"]="https://github.com/Artrajz/vits-simple-api"
)

echo -e "${PLAIN}${GREEN}Choose a language/选择语言: ${PLAIN}"
echo "1. English"
echo "2. 中文"
read -p "Enter your choice (1 or 2): " choice_language

declare -A MESSAGES
if [ "$choice_language" -eq 1 ]; then
  for key in "${!EN_MESSAGES[@]}"; do
    MESSAGES["$key"]="${EN_MESSAGES[$key]}"
  done
else
  for key in "${!ZH_MESSAGES[@]}"; do
    MESSAGES["$key"]="${ZH_MESSAGES[$key]}"
  done
fi

mkdir -p $INSTALL_DIR
cd $INSTALL_DIR

download_with_fallback() {
  local filename=$1
  shift # Shift arguments to the left to handle URLs

  local success=0
  local url
  for url in "$@"; do
    echo -e "${YELLOW}${MESSAGES["ATTEMPT_DOWNLOAD"]} $filename ${MESSAGES["FROM"]} $url\n${PLAIN}"
    if wget -O "$INSTALL_DIR/$filename" "$url"; then
      success=1
      break
    fi
  done

  if [ "$success" -ne 1 ]; then
    echo -e "${RED} $filename ${MESSAGES["FROM_ALL_URLS"]} ${MESSAGES["DOWNLOAD_FAIL"]}${PLAIN}"
    exit 1
  fi
}

version_gt() {
  test "$(echo "$@" | tr " " "\n" | sort -V | head -n 1)" != "$1"
}

while true; do
  echo -e "${GREEN}${MESSAGES["CHOOSE_VERSION"]}${PLAIN}"
  echo -e "1. ${MESSAGES["DOCKER_CPU"]}"
  echo -e "2. ${MESSAGES["DOCKER_GPU"]}"
  read -p "${MESSAGES["ENTER_CHOICE"]}" choice_gpu
  case $choice_gpu in
  1)
    echo -e "${MESSAGES["DOWNLOADING"]}"
    download_with_fallback docker-compose.yaml \
      "https://raw.githubusercontent.com/Artrajz/vits-simple-api/main/docker-compose.yaml" \
      "https://ghproxy.com/https://raw.githubusercontent.com/Artrajz/vits-simple-api/main/docker-compose.yaml"
    break
    ;;
  2)
    echo -e "${MESSAGES["DOWNLOADING"]}"
    download_with_fallback docker-compose.yaml \
      "https://raw.githubusercontent.com/Artrajz/vits-simple-api/main/docker-compose-gpu.yaml" \
      "https://ghproxy.com/https://raw.githubusercontent.com/Artrajz/vits-simple-api/main/docker-compose-gpu.yaml"
    break
    ;;
  *)
    echo -e "${RED}${MESSAGES["INVALID_CHOICE"]}${PLAIN}"
    ;;
  esac
done

if [ "$choice_gpu" -eq 2 ]; then
  DOCKER_VERSION=$(docker version --format '{{.Server.Version}}')
  MIN_DOCKER_VERSION="19.03"

  if version_gt $MIN_DOCKER_VERSION $DOCKER_VERSION; then
    echo -e "${RED}Your Docker version ($DOCKER_VERSION) does not support GPU. You need at least version $MIN_DOCKER_VERSION.${PLAIN}"
    exit 1
  fi
fi

if ! command -v docker-compose &>/dev/null; then
  echo -e "${RED}docker-compose could not be found.${PLAIN}"
  exit 1
fi

echo -e "${GREEN}${MESSAGES["PULL_IMAGE"]}${PLAIN}"
read -p "${MESSAGES["ENTER_CHOICE"]}" choice_pull

if [ "$choice_pull" -eq 1 ]; then
  docker compose pull
  docker compose up -d
fi

echo -e "${YELLOW}${MESSAGES["DOWNLOAD_CONFIG"]}${PLAIN}"

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

download_with_fallback config.example.py \
  "https://raw.githubusercontent.com/Artrajz/vits-simple-api/main/config.py" \
  "https://ghproxy.com/https://raw.githubusercontent.com/Artrajz/vits-simple-api/main/config.py"

download_with_fallback gunicorn_config.example.py \
  "https://raw.githubusercontent.com/Artrajz/vits-simple-api/main/gunicorn_config.py" \
  "https://ghproxy.com/https://raw.githubusercontent.com/Artrajz/vits-simple-api/main/gunicorn_config.py"

echo -e "${GREEN}${MESSAGES["DOWNLOAD_DICT"]}${PLAIN}"
echo -e "${GREEN}${MESSAGES["MUST_DOWNLOAD_JP"]}${PLAIN}"
read -p "${MESSAGES["ENTER_CHOICE"]}" choice_download_pyopenjtalk

if [ "$choice_download_pyopenjtalk" -eq 1 ]; then
  mkdir -p pyopenjtalk
  echo -e "${MESSAGES["DOWNLOADING"]}"
  download_with_fallback open_jtalk_dic_utf_8-1.11.tar.gz \
    "https://github.com/r9y9/open_jtalk/releases/download/v1.11.1/open_jtalk_dic_utf_8-1.11.tar.gz" \
    "https://ghproxy.com/https://github.com/r9y9/open_jtalk/releases/download/v1.11.1/open_jtalk_dic_utf_8-1.11.tar.gz"
  echo -e "${MESSAGES["UNZIPPING"]}"
  tar -xzvf open_jtalk_dic_utf_8-1.11.tar.gz -C pyopenjtalk/
  rm open_jtalk_dic_utf_8-1.11.tar.gz
fi

echo -e "${GREEN}${MESSAGES["DOWNLOAD_VITS_CHINESE"]}${PLAIN}"
echo -e "${GREEN}${MESSAGES["MUST_DOWNLOAD_VITS_CHINESE"]}${PLAIN}"
read -p "${MESSAGES["ENTER_CHOICE"]}" choice_download_vits_chinese

if [ "$choice_download_vits_chinese" -eq 1 ]; then
  mkdir -p vits/bert

  EXPECTED_MD5="dea78034433141adc8002404aa1b3184"
  FILE_PATH="vits/bert/prosody_model.pt"
  echo -e "${MESSAGES["VERIFYING"]}$FILE_PATH"
  ACTUAL_MD5=$(md5sum $FILE_PATH | awk '{print $1}')

  if [ "$EXPECTED_MD5" == "$ACTUAL_MD5" ]; then
    echo "${MESSAGES["DOWNLOADED"]}"
  else
    echo "${MESSAGES["CORRUPTED"]}"
    download_with_fallback vits/bert/prosody_model.pt \
      "https://huggingface.co/spaces/maxmax20160403/vits_chinese/resolve/main/bert/prosody_model.pt"
  fi

fi

echo -e "${GREEN}${MESSAGES["DOWNLOAD_BERT_VITS2"]}${PLAIN}"
echo -e "${GREEN}${MESSAGES["MUST_DOWNLOAD_BERT_VITS2"]}${PLAIN}"
read -p "${MESSAGES["ENTER_CHOICE"]}" choice_download_bert_vits2

if [ "$choice_download_bert_vits2" -eq 1 ]; then
  mkdir -p bert_vits2/bert/chinese-roberta-wwm-ext-large

  EXPECTED_MD5="15d7435868fef1bd4222ff7820149a2a"
  FILE_PATH="bert_vits2/bert/chinese-roberta-wwm-ext-large/pytorch_model.bin"
  echo -e "${MESSAGES["VERIFYING"]}$FILE_PATH"
  ACTUAL_MD5=$(md5sum $FILE_PATH | awk '{print $1}')

  if [ "$EXPECTED_MD5" == "$ACTUAL_MD5" ]; then
    echo "${MESSAGES["DOWNLOADED"]}"
  else
    echo ${MESSAGES["CORRUPTED"]}
    download_with_fallback bert_vits2/bert/chinese-roberta-wwm-ext-large/pytorch_model.bin \
      "https://huggingface.co/hfl/chinese-roberta-wwm-ext-large/resolve/main/pytorch_model.bin"
  fi
  
  mkdir -p bert_vits2/bert/bert-base-japanese-v3

  EXPECTED_MD5="6d0f8f3503dae04df0711b6175ef0c8e"
  FILE_PATH="bert_vits2/bert/bert-base-japanese-v3/pytorch_model.bin"
  echo -e "${MESSAGES["VERIFYING"]}$FILE_PATH"
  ACTUAL_MD5=$(md5sum $FILE_PATH | awk '{print $1}')

  if [ "$EXPECTED_MD5" == "$ACTUAL_MD5" ]; then
    echo "${MESSAGES["DOWNLOADED"]}"
  else
    echo ${MESSAGES["CORRUPTED"]}
    download_with_fallback bert_vits2/bert/bert-base-japanese-v3/pytorch_model.bin \
      "https://huggingface.co/cl-tohoku/bert-base-japanese-v3/resolve/main/pytorch_model.bin"
  fi

fi

if [ "$choice_gpu" -eq 2 ]; then
  if ! docker run --gpus all artrajz/vits-simple-api:latest-gpu nvidia-smi &>/dev/null; then
    echo -e "${RED}Your Docker does not seem to support GPU or NVIDIA Docker is not installed properly.${PLAIN}"
    exit 1
  fi
fi

echo -e "\n${MESSAGES["INSTALL_COMPLETE"]}"
echo -e "${MESSAGES["CONFIG_DIR"]} $(realpath $INSTALL_DIR)"
echo -e "${YELLOW}${MESSAGES["IMPORT_NOTICE"]}${PLAIN}"
echo -e "${YELLOW}${MESSAGES["RESTART_NOTICE"]}${PLAIN}"
echo -e "${MESSAGES["ISSUE_NOTICE"]}"
echo -e "${MESSAGES["GITHUB_LINK"]}"
