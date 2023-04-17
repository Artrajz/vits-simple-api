<div class="title" align=center>
    <h1>vits-simple-api</h1>
	<div>Simply call the vits api</div>
    <br/>
    <br/>
    <p>
        <img src="https://img.shields.io/github/license/Artrajz/vits-simple-api">
    	<img src="https://img.shields.io/badge/python-3.9%7C3.10-green">
        <a href="https://hub.docker.com/r/artrajz/vits-simple-api">
            <img src="https://img.shields.io/docker/pulls/artrajz/vits-simple-api"></a>
    </p>
    <a href="https://github.com/Artrajz/vits-simple-api/blob/main/README.md">English</a>|<a href="https://github.com/Artrajz/vits-simple-api/blob/main/README_zh.md">中文文档</a>
    <br/>
</div>




# Feature

- VITS text-to-speech
- HuBert-soft VITS
- VITS voice conversion
- Support for loading multiple models
- Automatic language recognition and processing,support for custom language type range
- Customize default parameters
- Long text batch processing
- GPU accelerated inference

<details><summary>Update Logs</summary><pre><code>
<h2>2023.4.17</h2>
<span>Added the feature that the cleaner for a single language needs to be annotated to clean, and added GPU acceleration for inference, but the GPU inference environment needs to be manually installed.</span>
<h2>2023.4.12</h2>
<span>Renamed the project from MoeGoe-Simple-API to vits-simple-api, added support for batch processing of long texts, and added a segment threshold "max" for long texts.</span>
<h2>2023.4.7</h2>
<span>Added a configuration file to customize default parameters. This update requires manually updating config.py. See config.py for specific usage.</span>
<h2>2023.4.6</h2>
<span>Added the "auto" option for automatically recognizing the language of the text. Modified the default value of the "lang" parameter to "auto". Automatic recognition still has some defects, please choose manually.</span>
<span>Unified the POST request type as multipart/form-data.</span>
</code></pre></details>



demo：`https://api.artrajz.cn/py/voice?text=你好,こんにちは&id=142`

# Deploy

## Docker

### Docker image pull script

```
bash -c "$(wget -O- https://raw.githubusercontent.com/Artrajz/vits-simple-api/main/vits-simple-api-installer-latest.sh)"
```

- The image size is 5GB, and it will be 8GB after decompression. Please prepare enough disk space.
- After a successful pull, the vits model needs to be imported before use. Please follow the steps below to import the model.

### Download  VITS model

Put the model into `/usr/local/vits-simple-api/Model`

<details><summary>Folder structure</summary><pre><code>
├─g
│      config.json
│      G_953000.pth
│
├─louise
│      360_epochs.pth
│      config.json
│      hubert-soft-0d54a1f4.pt
│
├─Nene_Nanami_Rong_Tang
│      1374_epochs.pth
│      config.json
│
└─Zero_no_tsukaima
        1158_epochs.pth
        config.json
</code></pre></details>



### Modify model path

Modify in  `/usr/local/vits-simple-api/config.py` 

<details><summary>config.py</summary><pre><code>
For each model, the filling method is as follows 模型列表中每个模型的填写方法如下
example 示例:
MODEL_LIST = [
    #VITS
    [ABS_PATH+"/Model/Nene_Nanami_Rong_Tang/1374_epochs.pth", ABS_PATH+"/Model/Nene_Nanami_Rong_Tang/config.json"],
    [ABS_PATH+"/Model/Zero_no_tsukaima/1158_epochs.pth", ABS_PATH+"/Model/Zero_no_tsukaima/config.json"],
    [ABS_PATH+"/Model/g/G_953000.pth", ABS_PATH+"/Model/g/config.json"],
    #HuBert-VITS
    [ABS_PATH+"/Model/louise/360_epochs.pth", ABS_PATH+"/Model/louise/config.json", ABS_PATH+"/Model/louise/hubert-soft-0d54a1f4.pt"],
]
</code></pre></details>



### Startup

`docker compose up -d`

Or execute the pull script again

### Image update 

Run the docker image pull script again 

## Virtual environment deployment

### Clone

`git clone https://github.com/Artrajz/vits-simple-api.git`

###  Download python dependencies 

A python virtual environment is recommended，use python >= 3.9

`pip install -r requirements.txt`

Fasttext may not be installed on windows, you can install it with the following command,or download wheels [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#fasttext)

```
#python3.10 win_amd64
pip install https://github.com/Artrajz/archived/raw/main/fasttext/fasttext-0.9.2-cp310-cp310-win_amd64.whl
#python3.9 win_amd64
pip install https://github.com/Artrajz/archived/raw/main/fasttext/fasttext-0.9.2-cp39-cp39-win_amd64.whl
```

### Download  VITS model 

Put the model into `/path/to/vits-simple-api/Model`

<details><summary>Folder structure</summary><pre><code>
├─g
│      config.json
│      G_953000.pth
│
├─louise
│      360_epochs.pth
│      config.json
│      hubert-soft-0d54a1f4.pt
│
├─Nene_Nanami_Rong_Tang
│      1374_epochs.pth
│      config.json
│
└─Zero_no_tsukaima
        1158_epochs.pth
        config.json
</code></pre></details>

### Modify model path

Modify in  `/path/to/vits-simple-api/config.py` 

<details><summary>config.py</summary><pre><code>
For each model, the filling method is as follows 模型列表中每个模型的填写方法如下
example 示例:
MODEL_LIST = [
    #VITS
    [ABS_PATH+"/Model/Nene_Nanami_Rong_Tang/1374_epochs.pth", ABS_PATH+"/Model/Nene_Nanami_Rong_Tang/config.json"],
    [ABS_PATH+"/Model/Zero_no_tsukaima/1158_epochs.pth", ABS_PATH+"/Model/Zero_no_tsukaima/config.json"],
    [ABS_PATH+"/Model/g/G_953000.pth", ABS_PATH+"/Model/g/config.json"],
    #HuBert-VITS
    [ABS_PATH+"/Model/louise/360_epochs.pth", ABS_PATH+"/Model/louise/config.json", ABS_PATH+"/Model/louise/hubert-soft-0d54a1f4.pt"],
]
</code></pre></details>

### Startup

`python app.py`

# GPU accelerated

## Windows
### Install CUDA
Check the highest version of CUDA supported by your graphics card:
```
nvidia-smi
```
Taking CUDA 11.7 as an example, download it from the official website: https://developer.nvidia.com/cuda-11-7-0-download-archive?target_os=Windows&amp;target_arch=x86_64&amp;target_version=10&amp;target_type=exe_local
### Install GPU version of PyTorch
```
pip3 install torch torchvision torchaudio --index-url ![img](file:///C:\Users\Administrator\AppData\Roaming\Tencent\QQTempSys\%W@GJ$ACOF(TYDYECOKVDYB.png)https://download.pytorch.org/whl/cu117
```
You can find the corresponding command for the version you need on the official website: https://pytorch.org/get-started/locally/
## Linux
The installation process is similar, but I don't have the environment to test it.

# API

## GET

#### speakers list 

- GET http://127.0.0.1:23456/voice/speakers

  Returns the mapping table of role IDs to speaker names.

#### voice vits

- GET http://127.0.0.1/voice?text=text

  Default values are used when other parameters are not specified.

- GET http://127.0.0.1/voice?text=[ZH]text[ZH][JA]text[JA]&lang=mix

  When lang=mix, the text needs to be annotated.

- GET http://127.0.0.1/voice?text=text&id=142&format=wav&lang=zh&length=1.4

   The text is "text", the role ID is 142, the audio format is wav, the text language is zh, the speech length is 1.4, and the other parameters are default.

#### check

- GET http://127.0.0.1:23456/voice/check?id=0&model=vits

## POST

- python

```python
import re
import requests
import json
import os
import random
import string
from requests_toolbelt.multipart.encoder import MultipartEncoder

abs_path = os.path.dirname(__file__)
base = "http://127.0.0.1:23456"

#speakers
def voice_speakers():
    url = f"{base}/voice/speakers"

    res = requests.post(url=url)
    json = res.json()
    for i in json:
        print(i)
        for j in json[i]:
            print(j)

#voice vits
def voice_vits(text):
    fields = {
        "text":text,
        "id":"0",
        "format":"wav",
        "lang":"auto",
        "length":"1",
        "noise":"0.667",
        "noisew":"0.8",
        "max": "50"
    }
    boundary = '----VoiceConversionFormBoundary' \
               + ''.join(random.sample(string.ascii_letters + string.digits, 16))
    
    m = MultipartEncoder(fields=fields, boundary=boundary)
    headers = {"Content-Type": m.content_type}
    url = f"{base}/voice"

    res = requests.post(url=url,data=m,headers=headers)
    fname = re.findall("filename=(.+)", res.headers["Content-Disposition"])[0]
    path = f"{abs_path}/{fname}"
    
    with open(path, "wb") as f:
        f.write(res.content)
    print(path)

#hubert-vits
def voice_hubert_vits(upload_path):
    upload_name = os.path.basename(upload_path)
    upload_type = f'audio/{upload_name.split(".")[1]}' #wav,ogg
    
    with open(upload_path,'rb') as upload_file:
        fields = {
            "upload": (upload_name, upload_file,upload_type),
            "target_id":"0",
            "format":"wav",
            "length":"1",
            "noise":"0.1",
            "noisew":"0.1",
        }
        boundary = '----VoiceConversionFormBoundary' \
                   + ''.join(random.sample(string.ascii_letters + string.digits, 16))
        
        m = MultipartEncoder(fields=fields, boundary=boundary)
        headers = {"Content-Type": m.content_type}
        url = f"{base}/voice/hubert-vits"

        res = requests.post(url=url,data=m,headers=headers)
    fname = re.findall("filename=(.+)", res.headers["Content-Disposition"])[0]
    path = f"{abs_path}/{fname}"
    
    with open(path, "wb") as f:
        f.write(res.content)
    print(path)

#voice conversion in the same VITS model
def voice_conversion(upload_path):
    upload_name = os.path.basename(upload_path)
    upload_type = f'audio/{upload_name.split(".")[1]}' #wav,ogg

    with open(upload_path,'rb') as upload_file:
        fields = {
            "upload": (upload_name, upload_file,upload_type),
            "original_id": "3",
            "target_id": "0",
        }
        boundary = '----VoiceConversionFormBoundary' \
                   + ''.join(random.sample(string.ascii_letters + string.digits, 16))
        m = MultipartEncoder(fields=fields, boundary=boundary)
        
        headers = {"Content-Type": m.content_type}
        url = f"{base}/voice/conversion"

        res = requests.post(url=url,data=m,headers=headers)
        
    fname = re.findall("filename=(.+)", res.headers["Content-Disposition"])[0]
    path = f"{abs_path}/{fname}"
    
    with open(path, "wb") as f:
        f.write(res.content)
    print(path)
```

# Parameter

## voice vits

| Name                   | Parameter | Is must | Default | Type  | Instruction                                                  |
| ---------------------- | --------- | ------- | ------- | ----- | ------------------------------------------------------------ |
| Synthesized text       | text      | true    |         | str   |                                                              |
| Role ID                | id        | false   | 0       | int   |                                                              |
| Audio format           | format    | false   | wav     | str   | Support for wav,ogg,silk                                     |
| Text language          | lang      | false   | auto    | str   | The language of the text to be synthesized. Available options include auto, zh, ja, and mix. When lang=mix, the text should be wrapped in [ZH] or [JA].The default mode is auto, which automatically detects the language of the text |
| Audio length           | length    | false   | 1.0     | float | Adjusts the length of the synthesized speech, which is equivalent to adjusting the speed of the speech. The larger the value, the slower the speed. |
| Noise                  | noise     | false   | 0.667   | float |                                                              |
| Noise Weight           | noisew    | false   | 0.8     | float |                                                              |
| Segmentation threshold | max       | false   | 50      | int   |                                                              |

## voice conversion

| Name           | Parameter   | Is must | Default | Type | Instruction                                               |
| -------------- | ----------- | ------- | ------- | ---- | --------------------------------------------------------- |
| Uploaded Audio | upload      | true    |         | file | The audio file to be uploaded. It should be in wav or ogg |
| Source Role ID | original_id | true    |         | int  | The ID of the role used to upload the audio file.         |
| Target Role ID | target_id   | true    |         | int  | The ID of the target role to convert the audio to.        |

## HuBert-VITS

| Name           | Parameter | Is must | Default | Type  | Instruction                                                  |
| -------------- | --------- | ------- | ------- | ----- | ------------------------------------------------------------ |
| Uploaded Audio | upload    | true    |         | file  | he audio file to be uploaded. It should be in wav or ogg format. |
| Target Role ID | target_id | true    |         | int   |                                                              |
| Audio format   | format    | true    |         | str   | wav,ogg,silk                                                 |
| Audio length   | length    | true    |         | float | Adjusts the length of the synthesized speech, which is equivalent to adjusting the speed of the speech. The larger the value, the slower the speed. |
| Noise          | noise     | true    |         | float |                                                              |
| Noise Weight   | noisew    | true    |         | float |                                                              |

# Communication

Learning and communication,now there is only Chinese [QQ group](https://qm.qq.com/cgi-bin/qm/qr?k=-1GknIe4uXrkmbDKBGKa1aAUteq40qs_&jump_from=webapi&authKey=x5YYt6Dggs1ZqWxvZqvj3fV8VUnxRyXm5S5Kzntc78+Nv3iXOIawplGip9LWuNR/)

# Acknowledgements

- vits:https://github.com/jaywalnut310/vits
- MoeGoe:https://github.com/CjangCjengh/MoeGoe

