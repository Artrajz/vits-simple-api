INSTALL_DIR=/usr/local/vits-simple-api

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
PLAIN='\033[0m'

DOCKER_COMPOSE_URL=https://github.com/docker/compose/releases/download/v2.29.2/docker-compose-`uname -s`-`uname -m`

declare -A EN_MESSAGES
declare -A ZH_MESSAGES
declare -A JA_MESSAGES

EN_MESSAGES=(
  ["ERROR_NO_CURL"]="Error: curl not detected. Please install this program first."
  ["WARNING_NO_DOCKER"]="Warning: Docker not detected."
  ["WARNING_NO_DOCKER_COMPOSE"]="Warning: Docker Compose not detected."
  ["WARNING_NO_NVIDIA_TOOLKIT"]="Warning: nvidia-container-toolkit not installed."
  ["INSTALL_PROMPT"]="Do you want to install it automatically? Enter 1 for yes or 2 for no:"
  ["ENTER_Y_OR_N"]="Please enter y or n."
  ["ATTEMPT_DOWNLOAD"]="Attempting to download"
  ["FROM"]="from"
  ["DOWNLOAD_FAIL"]="Failed to download"
  ["FROM_ALL_URLS"]="from all provided URLs."
  ["DOWNLOADING"]="Downloading..."
  ["VERIFYING"]="Verifying..."
  ["UNZIPPING"]="Unzipping..."
  ["CREATE_PLACEHOLDER_FILE"]="Created placeholder file"
  ["CHOOSE_VERSION"]="Which version of docker-compose.yaml do you want to download?"
  ["DOCKER_CPU"]="docker-compose.yaml (CPU version)"
  ["DOCKER_GPU"]="docker-compose-gpu.yaml (GPU version)"
  ["NO_DOWNLOAD"]="The YAML file already exists, no download needed"
  ["ENTER_CHOICE"]="Please enter your choice: "
  ["INVALID_CHOICE"]="Invalid choice. Please enter 1, 2, or 3."
  ["DOWNLOAD_CONFIG"]="Downloading configuration file shortly..."
  ["PULL_IMAGE"]="Do you want to start pulling the image? Enter 1 for yes or 2 for no"
  ["DOWNLOAD_DICT"]="Do you want to download the pyopenjtalk dictionary file? Enter 1 for yes or 2 for no"
  ["MUST_DOWNLOAD_JP"]="Japanese model must be downloaded."
  ["DOWNLOADED"]="File is downloaded correctly."
  ["CORRUPTED"]="The file may not have been downloaded, or the download might be incomplete, and it could also be corrupted."
  ["INSTALL_COMPLETE"]="The upgrade or installation has been completed."
  ["CONFIG_DIR"]="The configuration file directory is"
  ["IMPORT_NOTICE"]="If the vits model is not imported, it cannot be used. Import the model in the configuration file directory."
  ["RESTART_NOTICE"]="After modifying the configuration file, restart the docker container for the modification to take effect."
  ["ISSUE_NOTICE"]="If you have any questions, please put them in the issues."
  ["GITHUB_LINK"]="https://github.com/Artrajz/vits-simple-api"
  ["START_CONTAINERS_PROMPT"]="Do you want to start the containers? Enter 1 for yes, 2 for no."
  ["CONTAINERS_STARTING"]="Container is starting"
)

ZH_MESSAGES=(
  ["ERROR_NO_CURL"]="错误：未检测到 curl，请先安装此程序。"
  ["WARNING_NO_DOCKER"]="警告：未检测到 Docker。"
  ["WARNING_NO_DOCKER_COMPOSE"]="警告：未检测到 Docker Compose。"
  ["WARNING_NO_NVIDIA_TOOLKIT"]="警告：nvidia-container-toolkit 未安装。"
  ["INSTALL_PROMPT"]="是否自动为你安装？输入1表示是，2表示否："
  ["ENTER_Y_OR_N"]="请输入 y 或 n。"
  ["ATTEMPT_DOWNLOAD"]="正在尝试下载"
  ["FROM"]="从"
  ["DOWNLOAD_FAIL"]="都下载失败"
  ["FROM_ALL_URLS"]="从所有提供的URLs"
  ["DOWNLOADING"]="正在下载..."
  ["VERIFYING"]="正在校验"
  ["UNZIPPING"]="正在解压..."
  ["CREATE_PLACEHOLDER_FILE"]="创建占位文件"
  ["CHOOSE_VERSION"]="你想下载哪个版本的docker-compose.yaml？"
  ["DOCKER_CPU"]="docker-compose.yaml (CPU版本)"
  ["DOCKER_GPU"]="docker-compose-gpu.yaml (GPU版本)"
  ["NO_DOWNLOAD"]="已有yaml文件，不下载"
  ["ENTER_CHOICE"]="请输入您的选择: "
  ["INVALID_CHOICE"]="无效的选择，请输入1、2或3。"
  ["DOWNLOAD_CONFIG"]="即将下载配置文件..."
  ["PULL_IMAGE"]="是否要开始拉取镜像？输入1表示是，2表示否。"
  ["DOWNLOAD_DICT"]="是否要下载pyopenjtalk的词典文件？输入1表示是，2表示否。"
  ["MUST_DOWNLOAD_JP"]="使用日语模型必须下载该词典文件，将占用大约102MB。"
  ["DOWNLOADED"]="文件已正确下载。"
  ["CORRUPTED"]="文件可能未下载，或下载不完整，也有可能已损坏。"
  ["INSTALL_COMPLETE"]="更新或安装已完成。"
  ["CONFIG_DIR"]="配置文件目录是"
  ["IMPORT_NOTICE"]="如果vits模型没有被导入，它是无法使用的。请在配置文件目录中导入模型。"
  ["RESTART_NOTICE"]="修改配置文件后，请重启docker容器以使修改生效。"
  ["ISSUE_NOTICE"]="如果你有任何问题，请在issues中提出，或者加入q群提问。"
  ["GITHUB_LINK"]="https://github.com/Artrajz/vits-simple-api"
  ["START_CONTAINERS_PROMPT"]="是否要启动容器？输入1表示是，2表示否。"
  ["CONTAINERS_STARTING"]="容器正在启动"
)

JA_MESSAGES=(
  ["ERROR_NO_CURL"]="エラー：curlが検出されませんでした。最初にこのプログラムをインストールしてください。"
  ["WARNING_NO_DOCKER"]="警告：Dockerが検出されませんでした。"
  ["WARNING_NO_DOCKER_COMPOSE"]="警告：Docker Compose が検出されませんでした。"
  ["WARNING_NO_NVIDIA_TOOLKIT"]="警告：nvidia-container-toolkit がインストールされていません。"
  ["INSTALL_PROMPT"]="自動でインストールしますか？はいの場合は1を、いいえの場合は2を入力してください:"
  ["ENTER_Y_OR_N"]="y または n を入力してください。"
  ["ATTEMPT_DOWNLOAD"]="ダウンロードを試みています"
  ["FROM"]="から"
  ["DOWNLOAD_FAIL"]="ダウンロードに失敗しました"
  ["FROM_ALL_URLS"]="提供されたすべてのURLから。"
  ["DOWNLOADING"]="ダウンロード中..."
  ["VERIFYING"]="検証中..."
  ["UNZIPPING"]="解凍中..."
  ["CREATE_PLACEHOLDER_FILE"]="プレースホルダーファイルを作成しました"
  ["CHOOSE_VERSION"]="どのバージョンのdocker-compose.yamlをダウンロードしますか？"
  ["DOCKER_CPU"]="docker-compose.yaml (CPUバージョン)"
  ["DOCKER_GPU"]="docker-compose-gpu.yaml (GPUバージョン)"
  ["NO_DOWNLOAD"]="YAMLファイルは既に存在しているため、ダウンロードの必要はありません"
  ["ENTER_CHOICE"]="選択肢を入力してください: "
  ["INVALID_CHOICE"]="無効な選択です。1、2、または3を入力してください。"
  ["DOWNLOAD_CONFIG"]="まもなく設定ファイルをダウンロードします..."
  ["PULL_IMAGE"]="イメージのプルを開始しますか？はいの場合は1を、いいえの場合は2を入力してください"
  ["DOWNLOAD_DICT"]="pyopenjtalk辞書ファイルをダウンロードしますか？はいの場合は1を、いいえの場合は2を入力してください"
  ["MUST_DOWNLOAD_JP"]="日本語モデルをダウンロードする必要があります。"
  ["DOWNLOADED"]="ファイルが正しくダウンロードされました。"
  ["CORRUPTED"]="ファイルがダウンロードされていないか、ダウンロードが不完全である可能性があります。また、ファイルが破損している可能性もあります。"
  ["INSTALL_COMPLETE"]="アップグレードまたはインストールが完了しました。"
  ["CONFIG_DIR"]="設定ファイルのディレクトリは以下の通りです"
  ["IMPORT_NOTICE"]="vitsモデルがインポートされていない場合、使用できません。設定ファイルのディレクトリでモデルをインポートしてください。"
  ["RESTART_NOTICE"]="設定ファイルを変更した後、変更を反映させるためにdockerコンテナを再起動してください。"
  ["ISSUE_NOTICE"]="何か質問がある場合は、イシューに投稿してください。"
  ["GITHUB_LINK"]="https://github.com/Artrajz/vits-simple-api"
  ["START_CONTAINERS_PROMPT"]="コンテナを起動しますか？はいの場合は1を、いいえの場合は2を入力してください。"
  ["CONTAINERS_STARTING"]="コンテナが起動中です"
)

echo -e "${PLAIN}${GREEN}Choose a language/选择语言/言語: ${PLAIN}"
echo "1. English"
echo "2. 中文"
echo "3. 日本語"
read -p "Enter your choice (1 or 2 or 3): " choice_language

declare -A MESSAGES
if [ "$choice_language" -eq 1 ]; then
  for key in "${!EN_MESSAGES[@]}"; do
    MESSAGES["$key"]="${EN_MESSAGES[$key]}"
  done
elif [ "$choice_language" -eq 3 ]; then
  for key in "${!JA_MESSAGES[@]}"; do
    MESSAGES["$key"]="${JA_MESSAGES[$key]}"
  done
else
  for key in "${!ZH_MESSAGES[@]}"; do
    MESSAGES["$key"]="${ZH_MESSAGES[$key]}"
  done
fi

if ! [ -x "$(command -v curl)" ]; then
  echo -e "${MESSAGES["ERROR_NO_CURL"]}"
fi

if ! [ -x "$(command -v docker)" ]; then
  echo -e "${MESSAGES["WARNING_NO_DOCKER"]}"
  while true; do
    read -p "${MESSAGES["INSTALL_PROMPT"]}" choice_install_docker
    case $choice_install_docker in
        1 ) curl -fsSL https://get.docker.com -o get-docker.sh; sudo -E sh get-docker.sh; rm get-docker.sh; break;;
        2 ) exit 1;;
    esac
  done
fi

if ! [ -x "$(command -v docker-compose)" ]; then
  echo -e "${MESSAGES["WARNING_NO_DOCKER_COMPOSE"]}"
  while true; do
    read -p "${MESSAGES["INSTALL_PROMPT"]}" choice_install_docker_compose
    case $choice_install_docker_compose in
        1 ) sudo -E curl -L "${DOCKER_COMPOSE_URL}" -o /usr/local/bin/docker-compose; sudo -E chmod +x /usr/local/bin/docker-compose; break;;
        2 ) exit 1;;
    esac
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
    if wget --connect-timeout=10 -O "$INSTALL_DIR/$filename" "$url"; then
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

create_placeholder_files() {
  for file_path in "$@"; do
    directory=$(dirname "$file_path")
    mkdir -p "$directory"
    if [ -d "$file_path" ]; then
      rm -rf "$file_path"
    fi
    if [ ! -f "$file_path" ]; then
      touch "$file_path"
      echo "${MESSAGES["CREATE_PLACEHOLDER_FILE"]}: $file_path"
    fi
  done
}

create_placeholder_files "config.yaml"

while true; do
  echo -e "${GREEN}${MESSAGES["CHOOSE_VERSION"]}${PLAIN}"
  echo -e "1. ${MESSAGES["DOCKER_CPU"]}"
  echo -e "2. ${MESSAGES["DOCKER_GPU"]}"
  echo -e "3. ${MESSAGES["NO_DOWNLOAD"]}"
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

    if ! [ -x "$(command -v nvidia-container-toolkit)" ]; then
      echo -e "${MESSAGES["WARNING_NO_NVIDIA_TOOLKIT"]}"
      while true; do
        read -p "${MESSAGES["INSTALL_PROMPT"]}" choice_install_nvdia_toolkit
        case $choice_install_nvdia_toolkit in
            1 ) sudo apt-get update; sudo apt-get install -y nvidia-container-toolkit; break;;
            2 ) exit 1;;
        esac
      done
    fi
    break
    ;;
  3)
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
fi

echo -e "${YELLOW}${MESSAGES["DOWNLOAD_CONFIG"]}${PLAIN}"

download_with_fallback config.example.py \
  "https://raw.githubusercontent.com/Artrajz/vits-simple-api/main/config.py" \
  "https://ghproxy.com/https://raw.githubusercontent.com/Artrajz/vits-simple-api/main/config.py"

if [ ! -f config.py ]; then
  cp config.example.py config.py
fi

download_with_fallback gunicorn_config.example.py \
  "https://raw.githubusercontent.com/Artrajz/vits-simple-api/main/gunicorn_config.py" \
  "https://ghproxy.com/https://raw.githubusercontent.com/Artrajz/vits-simple-api/main/gunicorn_config.py"

if [ ! -f gunicorn_config.py ]; then
  cp gunicorn_config.example.py gunicorn_config.py
fi

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

if [ "$choice_gpu" -eq 2 ]; then
  if ! docker run --gpus all artrajz/vits-simple-api:latest-gpu nvidia-smi &>/dev/null; then
    echo -e "${RED}Your Docker does not seem to support GPU or NVIDIA Docker is not installed properly.${PLAIN}"
    exit 1
  fi
fi

read -p "${MESSAGES["START_CONTAINERS_PROMPT"]}" choice_start
if [ "$choice_start" -eq 1 ]; then
  echo -e "${MESSAGES["CONTAINERS_STARTING"]}"
  docker compose up -d
fi

echo -e "\n${MESSAGES["INSTALL_COMPLETE"]}"
echo -e "${MESSAGES["CONFIG_DIR"]} $(realpath $INSTALL_DIR)"
echo -e "${YELLOW}${MESSAGES["IMPORT_NOTICE"]}${PLAIN}"
echo -e "${YELLOW}${MESSAGES["RESTART_NOTICE"]}${PLAIN}"
echo -e "${MESSAGES["ISSUE_NOTICE"]}"
echo -e "${MESSAGES["GITHUB_LINK"]}"
