<div class="title" align=center>
    <h1>vits-simple-api</h1>
	<div>Simply call the vits api</div>
    <br/>
    <br/>
    <p>
        <img src="https://img.shields.io/github/license/Artrajz/vits-simple-api">
    	<img src="https://img.shields.io/badge/python-3.10-green">
        <a href="https://hub.docker.com/r/artrajz/vits-simple-api">
            <img src="https://img.shields.io/docker/pulls/artrajz/vits-simple-api"></a>
    </p>
    <a href="https://github.com/Artrajz/vits-simple-api/blob/main/README.md">English</a>|<a href="https://github.com/Artrajz/vits-simple-api/blob/main/README_zh.md">中文文档</a>
    <br/>
</div>





# Feature

- [x] VITS text-to-speech, voice conversion
- [x] HuBert-soft VITS
- [x] [vits_chinese](https://github.com/PlayVoice/vits_chinese)
- [x] [Bert-VITS2](https://github.com/Stardust-minus/Bert-VITS2)
- [x] W2V2 VITS / [emotional-vits](https://github.com/innnky/emotional-vits) dimensional emotion model
- [x] Support for loading multiple models
- [x] Automatic language recognition and processing,set the scope of language type recognition according to model's cleaner,support for custom language type range
- [x] Customize default parameters
- [x] Long text batch processing
- [x] GPU accelerated inference
- [x] SSML (Speech Synthesis Markup Language) work in progress...


## Online Demo

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/Artrajz/vits-simple-api) Thanks to Hugging Face!

Please note that different IDs may support different languages.[speakers](https://artrajz-vits-simple-api.hf.space/voice/speakers)

- `https://artrajz-vits-simple-api.hf.space/voice/vits?text=你好,こんにちは&id=164`
- `https://artrajz-vits-simple-api.hf.space/voice/vits?text=Difficult the first time, easy the second.&id=4`
- excited:`https://artrajz-vits-simple-api.hf.space/voice/w2v2-vits?text=こんにちは&id=3&emotion=111`
- whispered:`https://artrajz-vits-simple-api.hf.space/w2v2-vits?text=こんにちは&id=3&emotion=2077` 

https://user-images.githubusercontent.com/73542220/237995061-c1f25b4e-dd86-438a-9363-4bb1fe65b425.mov

# Deployment

There are two deployment options to choose from. Regardless of the option you select, you'll need to import the model after deployment to use the application.

## Docker Deployment (Recommended for Linux)

### Step 1: Pull the Docker Image

Run the following command to pull the Docker image. Follow the prompts in the script to choose the necessary files to download and pull the image:

```bash
bash -c "$(wget -O- https://raw.githubusercontent.com/Artrajz/vits-simple-api/main/vits-simple-api-installer-latest.sh)"
```

The default paths for project configuration files and model folders are `/usr/local/vits-simple-api/`.

### Step 2: Start

Run the following command to start the container:

```bash
docker-compose up -d
```

### Image Update

To update the image, run the following commands:

```bash
docker-compose pull
```

Then, restart the container:

```bash
docker-compose up -d
```

## Virtual Environment Deployment

### Step 1: Clone the Project

Clone the project repository using the following command:

```bash
git clone https://github.com/Artrajz/vits-simple-api.git
```

### Step 2: Install Python Dependencies

It's recommended to use a Python virtual environment. Run the following command to install the required Python dependencies:

```bash
pip install -r requirements.txt
```

### Step 3: Start

Run the following command to start the program:

```bash
python app.py
```

## Windows Quick Deployment Package

### Step 1: Download and Extract the Deployment Package

Go to the [releases page](https://github.com/Artrajz/vits-simple-api/releases) and download the latest deployment package. Extract the downloaded files.

### Step 2: Start

Run `start.bat` to launch the program.

## Model Loading

### Step 1: Download VITS Models
Download the VITS model files and place them in the Model directory.

### Step 2: Loading Models

If you are starting for the first time, modify the default model path configuration in the config.py file (optional).

After the first startup, a config.yml configuration file will be generated. You can either modify the model_list in the configuration file or make changes through the admin backend in the browser.

You can specify the model paths using either absolute or relative paths, where relative paths are considered from the Model folder in the project's root directory.

For example, if the Model folder contains the following files:

```
├─model1
│  │─G_1000.pth
│  └─config.json
└─model2
   │─G_1000.pth
   └─config.json
```

You have multiple options for specifying the paths based on your preference.

Option 1:

```
'model_config':
  'model_list': 
  - - model1/G_1000.pth
    - model1/config.json
  - - model2/G_1000.pth
    - model2/config.json
```

Option 2:

```
'model_config':
  'model_list': 
  - [model1/G_1000.pth, model1/config.json]
  - [model2/G_1000.pth, model2/config.json]
```

Option 3:

```
'model_config':
  'model_list': [
    [model1/G_1000.pth, model1/config.json],
    [model2/G_1000.pth, model2/config.json],
  ]
```

# GPU accelerated

## Windows
### Install CUDA
Check the highest version of CUDA supported by your graphics card:
```
nvidia-smi
```
Taking CUDA 11.7 as an example, download it from the [official website](https://developer.nvidia.com/cuda-11-7-0-download-archive?target_os=Windows&amp;target_arch=x86_64&amp;target_version=10&amp;target_type=exe_local)
### Install GPU version of PyTorch

1.13.1+cu117 is recommended, other versions may have memory instability issues.

```
pip install torch==1.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
```
## Linux
The installation process is similar, but I don't have the environment to test it.

# Function Options Explanation

## Disable the Admin Backend

The admin backend allows loading and unloading models, and while it has login authentication, for added security, you can disable the admin backend in the `config.yml`:

```yaml
'IS_ADMIN_ENABLED': !!bool 'false'
```

This extra measure helps ensure absolute security when making the admin backend inaccessible to the public network.

## Bert-VITS2 Configuration and Language/Bert Model Usage

Starting from Bert-VITS2 v2.0, a model requires loading three different language Bert models. If you only need to use one or two languages, you can add the `lang` parameter in the `config.json` file of the model's data section. The value `['zh']` indicates that the model only uses Chinese and will load Chinese Bert models. The value `['zh', 'ja']` indicates the usage of both Chinese and Japanese bilingual models, and only Chinese and Japanese Bert models will be loaded. Similarly, this pattern continues for other language combinations.

Example:

```json
"data": {
  "lang": ["zh", "ja"],
  "training_files": "filelists/train.list",
  "validation_files": "filelists/val.list",
  "max_wav_value": 32768.0,
  ...
```

# Frequently Asked Questions

## Installation Issues with fastText Dependency

Fasttext may not be installed on windows, you can install it with the following command,or download wheels [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#fasttext)

```bash
# For Python 3.10 on win_amd64
pip install https://github.com/Artrajz/archived/raw/main/fasttext/fasttext-0.9.2-cp310-cp310-win_amd64.whl
```

or

```bash
pip install fasttext -i https://pypi.artrajz.cn/simple
```

## Installation Issues with pyopenjtalk Dependency

Since pypi.org does not provide a wheel file for pyopenjtalk, you often need to install it from the source code. This process might be cumbersome for some users, so you can also install it using a pre-built wheel as follows:

```bash
pip install pyopenjtalk -i https://pypi.artrajz.cn/simple
```



## Bert-VITS2 Version Compatibility

To ensure compatibility with the Bert-VITS2 model, modify the config.json file by adding a version parameter "version": "x.x.x". For instance, if the model version is 1.0.1, the configuration file should be written as:

```json
{
  "version": "1.0.1",
  "train": {
    "log_interval": 10,
    "eval_interval": 100,
    "seed": 52,
    ...
```

# Admin Backend
The default address is http://127.0.0.1:23456/admin.

The initial username and password can be found at the bottom of the config.yml file after the first startup.

# API

## GET

#### speakers list 

- GET http://127.0.0.1:23456/voice/speakers

  Returns the mapping table of role IDs to speaker names.

#### voice vits

- GET http://127.0.0.1:23456/voice/vits?text=text

  Default values are used when other parameters are not specified.

- GET http://127.0.0.1:23456/voice/vits?text=[ZH]text[ZH][JA]text[JA]&lang=mix

  When lang=mix, the text needs to be annotated.

- GET http://127.0.0.1:23456/voice/vits?text=text&id=142&format=wav&lang=zh&length=1.4

   The text is "text", the role ID is 142, the audio format is wav, the text language is zh, the speech length is 1.4, and the other parameters are default.

#### check

- GET http://127.0.0.1:23456/voice/check?id=0&model=vits

## POST

- See `api_test.py`

## API KEY

Set `API_KEY_ENABLED = True` in `config.py` to enable API key authentication. The API key is `API_KEY = "api-key"`.
After enabling it, you need to add the `api_key` parameter in GET requests and add the `X-API-KEY` parameter in the header for POST requests.

# Parameter

## VITS

| Name               | Parameter    | Is must | Default           | Type  | Instruction                                                  |
| ------------------ | ------------ | ------- | ----------------- | ----- | ------------------------------------------------------------ |
| Synthesized text   | text         | true    |                   | str   | Text needed for voice synthesis.                             |
| Speaker ID         | id           | false   | From `config.yml` | int   | The speaker ID.                                              |
| Audio format       | format       | false   | From `config.yml` | str   | Support for wav,ogg,silk,mp3,flac                            |
| Text language      | lang         | false   | From `config.yml` | str   | The language of the text to be synthesized. Available options include auto, zh, ja, and mix. When lang=mix, the text should be wrapped in [ZH] or [JA].The default mode is auto, which automatically detects the language of the text |
| Audio length       | length       | false   | From `config.yml` | float | Adjusts the length of the synthesized speech, which is equivalent to adjusting the speed of the speech. The larger the value, the slower the speed. |
| Noise              | noise        | false   | From `config.yml` | float | Sample noise, controlling the randomness of the synthesis.   |
| SDP noise          | noisew       | false   | From `config.yml` | float | Stochastic Duration Predictor noise, controlling the length of phoneme pronunciation. |
| Segment Size       | segment_size | false   | From `config.yml` | int   | Divide the text into paragraphs based on punctuation marks, and combine them into one paragraph when the length exceeds segment_size. If segment_size<=0, the text will not be divided into paragraphs. |
| Streaming response | streaming    | false   | false             | bool  | Streamed synthesized speech with faster initial response.    |

## VITS voice conversion

| Name           | Parameter   | Is must | Default | Type | Instruction                                               |
| -------------- | ----------- | ------- | ------- | ---- | --------------------------------------------------------- |
| Uploaded Audio | upload      | true    |         | file | The audio file to be uploaded. It should be in wav or ogg |
| Source Role ID | original_id | true    |         | int  | The ID of the role used to upload the audio file.         |
| Target Role ID | target_id   | true    |         | int  | The ID of the target role to convert the audio to.        |

## HuBert-VITS

| Name              | Parameter | Is must | Default | Type  | Instruction                                                  |
| ----------------- | --------- | ------- | ------- | ----- | ------------------------------------------------------------ |
| Uploaded Audio    | upload    | true    |         | file  | The audio file to be uploaded. It should be in wav or ogg format. |
| Target speaker ID | id        | true    |         | int   | The target  speaker ID.                                      |
| Audio format      | format    | true    |         | str   | wav,ogg,silk                                                 |
| Audio length      | length    | true    |         | float | Adjusts the length of the synthesized speech, which is equivalent to adjusting the speed of the speech. The larger the value, the slower the speed. |
| Noise             | noise     | true    |         | float | Sample noise, controlling the randomness of the synthesis.   |
| sdp noise         | noisew    | true    |         | float | Stochastic Duration Predictor noise, controlling the length of phoneme pronunciation. |

## W2V2-VITS

| Name                | Parameter    | Is must | Default           | Type  | Instruction                                                  |
| ------------------- | ------------ | ------- | ----------------- | ----- | ------------------------------------------------------------ |
| Synthesized text    | text         | true    |                   | str   | Text needed for voice synthesis.                             |
| Speaker ID          | id           | false   | From `config.yml` | int   | The speaker ID.                                              |
| Audio format        | format       | false   | From `config.yml` | str   | Support for wav,ogg,silk,mp3,flac                            |
| Text language       | lang         | false   | From `config.yml` | str   | The language of the text to be synthesized. Available options include auto, zh, ja, and mix. When lang=mix, the text should be wrapped in [ZH] or [JA].The default mode is auto, which automatically detects the language of the text |
| Audio length        | length       | false   | From `config.yml` | float | Adjusts the length of the synthesized speech, which is equivalent to adjusting the speed of the speech. The larger the value, the slower the speed. |
| Noise               | noise        | false   | From `config.yml` | float | Sample noise, controlling the randomness of the synthesis.   |
| SDP noise           | noisew       | false   | From `config.yml` | float | Stochastic Duration Predictor noise, controlling the length of phoneme pronunciation. |
| Segment Size        | segment_size | false   | From `config.yml` | int   | Divide the text into paragraphs based on punctuation marks, and combine them into one paragraph when the length exceeds segment_size. If segment_size<=0, the text will not be divided into paragraphs. |
| Dimensional emotion | emotion      | false   | 0                 | int   | The range depends on the emotion reference file in npy format, such as the  range of the [innnky](https://huggingface.co/spaces/innnky/nene-emotion/tree/main)'s model all_emotions.npy, which is 0-5457. |

## Dimensional emotion

| Name           | Parameter | Is must | Default | Type | Instruction                                                  |
| -------------- | --------- | ------- | ------- | ---- | ------------------------------------------------------------ |
| Uploaded Audio | upload    | true    |         | file | Return the npy file that stores the dimensional emotion vectors. |

## Bert-VITS2

| Name             | Parameter       | Is must | Default           | Type  | Instruction                                                  |
| ---------------- | --------------- | ------- | ----------------- | ----- | ------------------------------------------------------------ |
| Synthesized text | text            | true    |                   | str   | Text needed for voice synthesis.                             |
| Speaker ID       | id              | false   | From `config.yml` | int   | The speaker ID.                                              |
| Audio format     | format          | false   | From `config.yml` | str   | Support for wav,ogg,silk,mp3,flac                            |
| Text language    | lang            | false   | From `config.yml` | str   | "Auto" is a mode for automatic language detection and is also the default mode. However, it currently only supports detecting the language of an entire text passage and cannot distinguish languages on a per-sentence basis. The other available language options are "zh" and "ja". |
| Audio length     | length          | false   | From `config.yml` | float | Adjusts the length of the synthesized speech, which is equivalent to adjusting the speed of the speech. The larger the value, the slower the speed. |
| Noise            | noise           | false   | From `config.yml` | float | Sample noise, controlling the randomness of the synthesis.   |
| SDP noise        | noisew          | false   | From `config.yml` | float | Stochastic Duration Predictor noise, controlling the length of phoneme pronunciation. |
| Segment Size     | segment_size    | false   | From `config.yml` | int   | Divide the text into paragraphs based on punctuation marks, and combine them into one paragraph when the length exceeds segment_size. If segment_size<=0, the text will not be divided into paragraphs. |
| SDP/DP mix ratio | sdp_ratio       | false   | From `config.yml` | int   | The theoretical proportion of SDP during synthesis, the higher the ratio, the larger the variance in synthesized voice tone. |
| Emotion          | emotion         | false   | None              |       | Available for Bert-VITS2 v2.1, ranging from 0 to 9           |
| Reference Audio  | reference_audio | false   | None              |       | Available for Bert-VITS2 v2.1                                |


## SSML (Speech Synthesis Markup Language)

Supported Elements and Attributes

`speak` Element

| Attribute    | Instruction                                                  | Is must |
| ------------ | ------------------------------------------------------------ | ------- |
| id           | Default value is retrieved From `config.yml`                 | false   |
| lang         | Default value is retrieved From `config.yml`                 | false   |
| length       | Default value is retrieved From `config.yml`                 | false   |
| noise        | Default value is retrieved From `config.yml`                 | false   |
| noisew       | Default value is retrieved From `config.yml`                 | false   |
| segment_size | Splits text into segments based on punctuation marks. When the sum of segment lengths exceeds `segment_size`, it is treated as one segment. `segment_size<=0` means no segmentation. The default value is 0. | false   |
| model_type   | Default is VITS. Options: W2V2-VITS, BERT-VITS2              | false   |
| emotion      | Only effective when using W2V2-VITS . The range depends on the npy emotion reference file. | false   |
| sdp_ratio    | Only effective when using BERT-VITS2 .                       | false   |

`voice` Element

Higher priority than `speak`.

| Attribute    | Instruction                                                  | Is must |
| ------------ | ------------------------------------------------------------ | ------- |
| id           | Default value is retrieved From `config.yml`                 | false   |
| lang         | Default value is retrieved From `config.yml`                 | false   |
| length       | Default value is retrieved From `config.yml`                 | false   |
| noise        | Default value is retrieved From `config.yml`                 | false   |
| noisew       | Default value is retrieved From `config.yml`                 | false   |
| segment_size | Splits text into segments based on punctuation marks. When the sum of segment lengths exceeds `segment_size`, it is treated as one segment. `segment_size<=0` means no segmentation. The default value is 0. | false   |
| model_type   | Default is VITS. Options: W2V2-VITS, BERT-VITS2              | false   |
| emotion      | Only effective when using W2V2-VITS . The range depends on the npy emotion reference file. | false   |
| sdp_ratio    | Only effective when using BERT-VITS2 .                       | false   |

`break` Element

| Attribute | Instruction                                                  | Is must |
| --------- | ------------------------------------------------------------ | ------- |
| strength  | x-weak, weak, medium (default), strong, x-strong             | false   |
| time      | The absolute duration of a pause in seconds (such as `2s`) or milliseconds (such as `500ms`). Valid values range from 0 to 5000 milliseconds. If you set a value greater than the supported maximum, the service will use `5000ms`. If the `time` attribute is set, the `strength` attribute is ignored. | false   |

| Strength | Relative Duration |
| :------- | :---------------- |
| x-weak   | 250 ms            |
| weak     | 500 ms            |
| medium   | 750 ms            |
| strong   | 1000 ms           |
| x-strong | 1250 ms           |

Example

See `api_test.py`

# Communication

Learning and communication,now there is only Chinese [QQ group](https://qm.qq.com/cgi-bin/qm/qr?k=-1GknIe4uXrkmbDKBGKa1aAUteq40qs_&jump_from=webapi&authKey=x5YYt6Dggs1ZqWxvZqvj3fV8VUnxRyXm5S5Kzntc78+Nv3iXOIawplGip9LWuNR/)

# Acknowledgements

- vits:https://github.com/jaywalnut310/vits
- MoeGoe:https://github.com/CjangCjengh/MoeGoe
- emotional-vits:https://github.com/innnky/emotional-vits
- vits-uma-genshin-honkai:https://huggingface.co/spaces/zomehwh/vits-uma-genshin-honkai
- vits_chinese:https://github.com/PlayVoice/vits_chinese
- Bert_VITS2:https://github.com/fishaudio/Bert-VITS2

# Thank You to All Contributors

<a href="https://github.com/artrajz/vits-simple-ap/graphs/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=artrajz/vits-simple-api"/></a>
