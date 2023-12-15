INSTALL_DIR=/usr/local/vits-simple-api

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
PLAIN='\033[0m'

declare -A EN_MESSAGES
declare -A ZH_MESSAGES
declare -A JA_MESSAGES

EN_MESSAGES=(
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
  ["DOWNLOAD_VITS_CHINESE"]="Do you want to download the bert model for vits_chinese? Enter 1 for yes, 2 for no."
  ["MUST_DOWNLOAD_VITS_CHINESE"]="Using vits_chinese requires downloading these models, which will take up about 410MB."
  ["DOWNLOAD_BERT_VITS2_1"]="Do you want to download chinese-roberta-wwm-ext-large? This model is a Chinese BERT model used for the full version. It will occupy approximately 1.21GB. Enter 1 for yes, and 2 for no."
  ["DOWNLOAD_BERT_VITS2_2"]="Do you want to download bert-base-japanese-v3? This model is a Japanese BERT model used before version 2.0. It will occupy approximately 426MB. Enter 1 for yes, and 2 for no."
  ["DOWNLOAD_BERT_VITS2_3"]="Do you want to download bert-large-japanese-v2? Enter 1 for yes, and 2 for no."
  ["DOWNLOAD_BERT_VITS2_4"]="Do you want to download deberta-v2-large-japanese? This model is a Japanese BERT model used after version 2.0. It will occupy approximately 1.38GB. Enter 1 for yes, and 2 for no."
  ["DOWNLOAD_BERT_VITS2_5"]="Do you want to download deberta-v3-large? This model is an English BERT model used after version 2.0. It will occupy approximately 835MB. Enter 1 for yes, and 2 for no."
  ["MUST_DOWNLOAD_BERT_VITS2"]="To use Bert-VITS2, you must download these models, which will take up about 1.63GB."
  ["DOWNLOADED"]="File is downloaded correctly."
  ["CORRUPTED"]="The file may not have been downloaded, or the download might be incomplete, and it could also be corrupted."
  ["INSTALL_COMPLETE"]="The upgrade or installation has been completed."
  ["CONFIG_DIR"]="The configuration file directory is"
  ["IMPORT_NOTICE"]="If the vits model is not imported, it cannot be used. Import the model in the configuration file directory."
  ["RESTART_NOTICE"]="After modifying the configuration file, restart the docker container for the modification to take effect."
  ["ISSUE_NOTICE"]="If you have any questions, please put them in the issues."
  ["GITHUB_LINK"]="https://github.com/Artrajz/vits-simple-api"
  ["CONTAINERS_STARTING"]="Container is starting"
)

ZH_MESSAGES=(
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
  ["DOWNLOAD_VITS_CHINESE"]="是否要下载vits_chinese的bert模型？输入1表示是，2表示否。"
  ["MUST_DOWNLOAD_VITS_CHINESE"]="使用vits_chinese必须下载这些模型，将占用大约410MB。"
  ["DOWNLOAD_BERT_VITS2_1"]="是否要下载chinese-roberta-wwm-ext-large？该模型为全版本使用的中文bert模型。将占用大约1.21GB。输入1表示是，2表示否。"
  ["DOWNLOAD_BERT_VITS2_2"]="是否要下载bert-base-japanese-v3？该模型为2.0之前使用的日文bert模型。将占用大约426MB。输入1表示是，2表示否。"
  ["DOWNLOAD_BERT_VITS2_3"]="是否要下载bert-large-japanese-v2？输入1表示是，2表示否。"
  ["DOWNLOAD_BERT_VITS2_4"]="是否要下载deberta-v2-large-japanese？该模型为2.0以后使用的的日文bert模型。将占用大约1.38GB。输入1表示是，2表示否。"
  ["DOWNLOAD_BERT_VITS2_5"]="是否要下载deberta-v3-large？该模型为2.0以后使用的的英文文bert模型。将占用大约835MB。输入1表示是，2表示否。"
  ["MUST_DOWNLOAD_BERT_VITS2"]="使用Bert-VITS2必须下载这些模型，将占用大约1.63GB。"
  ["DOWNLOADED"]="文件已正确下载。"
  ["CORRUPTED"]="文件可能未下载，或下载不完整，也有可能已损坏。"
  ["INSTALL_COMPLETE"]="更新或安装已完成。"
  ["CONFIG_DIR"]="配置文件目录是"
  ["IMPORT_NOTICE"]="如果vits模型没有被导入，它是无法使用的。请在配置文件目录中导入模型。"
  ["RESTART_NOTICE"]="修改配置文件后，请重启docker容器以使修改生效。"
  ["ISSUE_NOTICE"]="如果你有任何问题，请在issues中提出，或者加入q群提问。"
  ["GITHUB_LINK"]="https://github.com/Artrajz/vits-simple-api"
  ["CONTAINERS_STARTING"]="容器正在启动"
)

JA_MESSAGES=(
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
	["DOWNLOAD_VITS_CHINESE"]="vits_chinese用のbertモデルをダウンロードしますか？はいの場合は1を、いいえの場合は2を入力してください。"
	["MUST_DOWNLOAD_VITS_CHINESE"]="vits_chineseを使用するには、これらのモデルをダウンロードする必要があり、約410MBの容量が必要です。"
	["DOWNLOAD_BERT_VITS2_1"]="chinese-roberta-wwm-ext-largeをダウンロードしますか？このモデルはフルバージョン用の中国語BERTモデルで、約1.21GBの容量を占めます。はいの場合は1を、いいえの場合は2を入力してください。"
	["DOWNLOAD_BERT_VITS2_2"]="bert-base-japanese-v3をダウンロードしますか？このモデルはバージョン2.0以前に使用される日本語BERTモデルで、約426MBの容量を占めます。はいの場合は1を、いいえの場合は2を入力してください。"
	["DOWNLOAD_BERT_VITS2_3"]="bert-large-japanese-v2をダウンロードしますか？はいの場合は1を、いいえの場合は2を入力してください。"
	["DOWNLOAD_BERT_VITS2_4"]="deberta-v2-large-japaneseをダウンロードしますか？このモデルはバージョン2.0以降に使用される日本語BERTモデルで、約1.38GBの容量を占めます。はいの場合は1を、いいえの場合は2を入力してください。"
	["DOWNLOAD_BERT_VITS2_5"]="deberta-v3-largeをダウンロードしますか？このモデルはバージョン2.0以降に使用される英語BERTモデルで、約835MBの容量を占めます。はいの場合は1を、いいえの場合は2を入力してください。"
	["MUST_DOWNLOAD_BERT_VITS2"]="Bert-VITS2を使用するには、これらのモデルをダウンロードする必要があり、約1.63GBの容量が必要です。"
	["DOWNLOADED"]="ファイルが正しくダウンロードされました。"
	["CORRUPTED"]="ファイルがダウンロードされていないか、ダウンロードが不完全である可能性があります。また、ファイルが破損している可能性もあります。"
	["INSTALL_COMPLETE"]="アップグレードまたはインストールが完了しました。"
	["CONFIG_DIR"]="設定ファイルのディレクトリは以下の通りです"
	["IMPORT_NOTICE"]="vitsモデルがインポートされていない場合、使用できません。設定ファイルのディレクトリでモデルをインポートしてください。"
	["RESTART_NOTICE"]="設定ファイルを変更した後、変更を反映させるためにdockerコンテナを再起動してください。"
	["ISSUE_NOTICE"]="何か質問がある場合は、イシューに投稿してください。"
	["GITHUB_LINK"]="https://github.com/Artrajz/vits-simple-api"
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

create_placeholder_files "config.yml" \
  "bert_vits2/bert/bert-base-japanese-v3/pytorch_model.bin" \
  "bert_vits2/bert/bert-large-japanese-v2/pytorch_model.bin" \
  "bert_vits2/bert/chinese-roberta-wwm-ext-large/pytorch_model.bin" \
  "bert_vits2/bert/deberta-v2-large-japanese/pytorch_model.bin" \
  "bert_vits2/bert/deberta-v3-large/pytorch_model.bin" \
  "bert_vits2/bert/deberta-v3-large/spm.model" \
  "vits/bert/prosody_model.pt"

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
      "https://huggingface.co/spaces/maxmax20160403/vits_chinese/resolve/main/bert/prosody_model.pt" \
      "https://hf-mirror.com/spaces/maxmax20160403/vits_chinese/resolve/main/bert/prosody_model.pt"
  fi

fi

echo -e "${GREEN}${MESSAGES["DOWNLOAD_BERT_VITS2_1"]}${PLAIN}"
read -p "${MESSAGES["ENTER_CHOICE"]}" choice_download_bert_vits2_1

if [ "$choice_download_bert_vits2_1" -eq 1 ]; then
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
      "https://huggingface.co/hfl/chinese-roberta-wwm-ext-large/resolve/main/pytorch_model.bin" \
      "https://hf-mirror.com/hfl/chinese-roberta-wwm-ext-large/resolve/main/pytorch_model.bin"
  fi
fi

echo -e "${GREEN}${MESSAGES["DOWNLOAD_BERT_VITS2_2"]}${PLAIN}"
read -p "${MESSAGES["ENTER_CHOICE"]}" choice_download_bert_vits2_2

if [ "$choice_download_bert_vits2_2" -eq 1 ]; then
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
      "https://huggingface.co/cl-tohoku/bert-base-japanese-v3/resolve/main/pytorch_model.bin" \
      "https://hf-mirror.com/cl-tohoku/bert-base-japanese-v3/resolve/main/pytorch_model.bin"
  fi

fi

echo -e "${GREEN}${MESSAGES["DOWNLOAD_BERT_VITS2_4"]}${PLAIN}"
read -p "${MESSAGES["ENTER_CHOICE"]}" choice_download_bert_vits2_4

if [ "$choice_download_bert_vits2_4" -eq 1 ]; then
  mkdir -p bert_vits2/bert/deberta-v2-large-japanese

  EXPECTED_MD5="1AAB4BC5DA8B5354315378439AC5BFA7"
  FILE_PATH="bert_vits2/bert/deberta-v2-large-japanese/pytorch_model.bin"
  echo -e "${MESSAGES["VERIFYING"]}$FILE_PATH"
  ACTUAL_MD5=$(md5sum $FILE_PATH | awk '{print $1}')

  if [ "$EXPECTED_MD5" == "$ACTUAL_MD5" ]; then
    echo "${MESSAGES["DOWNLOADED"]}"
  else
    echo ${MESSAGES["CORRUPTED"]}
    download_with_fallback bert_vits2/bert/deberta-v2-large-japanese/pytorch_model.bin \
      "https://huggingface.co/ku-nlp/deberta-v2-large-japanese/resolve/main/pytorch_model.bin" \
      "https://hf-mirror.com/ku-nlp/deberta-v2-large-japanese/resolve/main/pytorch_model.bin"
  fi

fi

echo -e "${GREEN}${MESSAGES["DOWNLOAD_BERT_VITS2_5"]}${PLAIN}"
read -p "${MESSAGES["ENTER_CHOICE"]}" choice_download_bert_vits2_5

if [ "$choice_download_bert_vits2_5" -eq 1 ]; then
  mkdir -p bert_vits2/bert/deberta-v3-large

  EXPECTED_MD5="917265658911F15661869FC4C06BB23C"
  FILE_PATH="bert_vits2/bert/deberta-v3-large/pytorch_model.bin"
  echo -e "${MESSAGES["VERIFYING"]}$FILE_PATH"
  ACTUAL_MD5=$(md5sum $FILE_PATH | awk '{print $1}')

  if [ "$EXPECTED_MD5" == "$ACTUAL_MD5" ]; then
    echo "${MESSAGES["DOWNLOADED"]}"
  else
    echo ${MESSAGES["CORRUPTED"]}
    download_with_fallback bert_vits2/bert/deberta-v3-large/pytorch_model.bin \
      "https://huggingface.co/microsoft/deberta-v3-large/resolve/main/pytorch_model.bin" \
      "https://hf-mirror.com/microsoft/deberta-v3-large/resolve/main/pytorch_model.bin"
  fi

  EXPECTED_MD5="1613FCBF3B82999C187B09C9DB79B568"
  FILE_PATH="bert_vits2/bert/deberta-v3-large/spm.model"
  echo -e "${MESSAGES["VERIFYING"]}$FILE_PATH"
  ACTUAL_MD5=$(md5sum $FILE_PATH | awk '{print $1}')

  if [ "$EXPECTED_MD5" == "$ACTUAL_MD5" ]; then
    echo "${MESSAGES["DOWNLOADED"]}"
  else
    echo ${MESSAGES["CORRUPTED"]}
    download_with_fallback bert_vits2/bert/deberta-v3-large/spm.model \
      "https://huggingface.co/microsoft/deberta-v3-large/resolve/main/spm.model" \
      "https://hf-mirror.com/microsoft/deberta-v3-large/resolve/main/spm.model"
  fi

fi

if [ "$choice_gpu" -eq 2 ]; then
  if ! docker run --gpus all artrajz/vits-simple-api:latest-gpu nvidia-smi &>/dev/null; then
    echo -e "${RED}Your Docker does not seem to support GPU or NVIDIA Docker is not installed properly.${PLAIN}"
    exit 1
  fi
fi

if [ "$choice_pull" -eq 1 ]; then
  echo ${MESSAGES["CONTAINERS_STARTING"]}
  docker compose up -d
fi

echo -e "\n${MESSAGES["INSTALL_COMPLETE"]}"
echo -e "${MESSAGES["CONFIG_DIR"]} $(realpath $INSTALL_DIR)"
echo -e "${YELLOW}${MESSAGES["IMPORT_NOTICE"]}${PLAIN}"
echo -e "${YELLOW}${MESSAGES["RESTART_NOTICE"]}${PLAIN}"
echo -e "${MESSAGES["ISSUE_NOTICE"]}"
echo -e "${MESSAGES["GITHUB_LINK"]}"
