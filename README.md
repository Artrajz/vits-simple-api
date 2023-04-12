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
    <br/>
</div>



# Feature

- VITS text-to-speech 语音合成
- HuBert-soft VITS 语音转换
- VITS voice conversion 语音转换
- Support for loading multiple models 加载多模型
- Automatic language recognition and processing,support for user-defined language type range 自动识别语言并处理，支持自定义语言类型范围
- Customize default parameters 自定义默认参数
- Long text batch processing 长文本批处理

<details><summary>Update Logs</summary><pre><code>
<h2>2023.4.8</h2>
<span>项目由MoeGoe-Simple-API更名为vits-simple-api，支持长文本批处理，增加长文本分段阈值max</span>
<h2>2023.4.7</h2>
<span>增加配置文件可自定义默认参数，本次更新需要手动更新config.py，具体使用方法见config.py</span>
<h2>2023.4.6</h2>
<span>加入自动识别语种选项auto，lang参数默认修改为auto，自动识别仍有一定缺陷，请自行选择</span>
<span>统一POST请求类型为multipart/form-data</span>
</code></pre></details>




demo：`https://api.artrajz.cn/py/voice?text=你好,こんにちは&id=142`

# Deploy

## Docker

### docker image pull script 镜像拉取脚本

```
bash -c "$(wget -O- https://raw.githubusercontent.com/Artrajz/vits-simple-api/main/vits-simple-api-installer-latest.sh)"
```

- The image size is 5GB, and it will be 8GB after decompression. Please prepare enough disk space.
- After a successful pull, the vits model needs to be imported before use. Please follow the steps below to import the model.

### Download  VITS model 下载VITS模型

Put the model into `/usr/local/moegoe-simple-api/Model`

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



### Modify model path 修改模型路径

Modify in  `/usr/local/moegoe-simple-api/config.py` 

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



### 开始使用！

终端输入`docker compose up -d`

或再次执行拉取脚本

### Image update 镜像更新方法

Run the docker image pull script again 重新执行docker镜像拉取脚本即可

## Direct deployment

### 克隆项目

`git clone https://github.com/Artrajz/MoeGoe-Simple-API.git`

### Download  VITS model 下载VITS模型

Put the model into `/usr/local/moegoe-simple-api/Model`

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



### Modify model path 修改模型路径

Modify in  `/usr/local/moegoe-simple-api/config.py` 

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




###  Download python dependencies 下载python依赖

A python virtual environment is recommended，use python >= 3.9

`pip install -r requirements.txt`

Fasttext may not be installed on windows, you can install it with the following command,or download wheels [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#fasttext)

windows下可能安装不了fasttext,可以用以下命令安装，附[wheels下载地址](https://www.lfd.uci.edu/~gohlke/pythonlibs/#fasttext)

```
#python3.10 win_amd64
pip install https://github.com/Artrajz/archived/raw/main/fasttext/fasttext-0.9.2-cp310-cp310-win_amd64.whl
#python3.9 win_amd64
pip install https://github.com/Artrajz/archived/raw/main/fasttext/fasttext-0.9.2-cp39-cp39-win_amd64.whl
```

### Startup 启动

`python app.py`

# API

## GET

#### speakers list 

- GET http://127.0.0.1:23456/voice/speakers

  返回id对应角色的映射表

#### voice vits

- GET http://127.0.0.1/voice?text=text

  其他参数不指定时均为默认值

- GET http://127.0.0.1/voice?text=text&id=142&format=wav&lang=zh&length=1.4

  文本为text，角色id为142，音频格式为wav，文本语言为zh，语音长度为1.4，其余参数默认

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
addr = "http://127.0.0.1:23456"

#映射表
def voice_speakers():
    url = f"{addr}/voice/speakers"

    res = requests.post(url=url)
    json = res.json()
    for i in json:
        print(i)
        for j in json[i]:
            print(j)

#语音合成 voice vits
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

#语音转换 hubert-vits
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
        url = f"{addr}/voice/hubert-vits"

        res = requests.post(url=url,data=m,headers=headers)
    fname = re.findall("filename=(.+)", res.headers["Content-Disposition"])[0]
    path = f"{abs_path}/{fname}"
    
    with open(path, "wb") as f:
        f.write(res.content)
    print(path)

#语音转换 同VITS模型内角色之间的音色转换
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
        url = f"{addr}/voice/conversion"

        res = requests.post(url=url,data=m,headers=headers)
        
    fname = re.findall("filename=(.+)", res.headers["Content-Disposition"])[0]
    path = f"{abs_path}/{fname}"
    
    with open(path, "wb") as f:
        f.write(res.content)
    print(path)
```

# Parameter

## voice vits 语音合成

| Name          | Parameter | Is must | Default | Type  | Instruction                                                  |
| ------------- | --------- | ------- | ------- | ----- | ------------------------------------------------------------ |
| 合成文本      | text      | true    |         | str   |                                                              |
| 角色id        | id        | false   | 0       | int   |                                                              |
| 音频格式      | format    | false   | wav     | str   | wav,ogg,silk                                                 |
| 文本语言      | lang      | false   | auto    | str   | auto,zh,ja,mix.auto为自动识别语言模式（仅中日文），也是默认模式。lang=mix时，文本应该用[ZH] 或 [JA] 包裹, |
| 语音长度/语速 | length    | false   | 1.0     | float | 调节语音长度，相当于调节语速，该数值越大语速越慢             |
| 噪声          | noise     | false   | 0.667   | float |                                                              |
| 噪声偏差      | noisew    | false   | 0.8     | float |                                                              |
| 分段阈值      | max       | false   | 50      | int   |                                                              |

## voice conversion 语音转换

| Name       | Parameter   | Is must | Default | Type       | Instruction            |
| ---------- | ----------- | ------- | ------- | ---------- | ---------------------- |
| 上传音频   | upload      | true    |         | audio file | wav or ogg             |
| 源角色id   | original_id | true    |         | int        | 上传文件所使用的角色id |
| 目标角色id | target_id   | true    |         | int        | 要转换的目标角色id     |

## HuBert-VITS 语音转换

| Name          | Parameter | Is must | Default | Type       | Instruction                                      |
| ------------- | --------- | ------- | ------- | ---------- | ------------------------------------------------ |
| 上传音频      | upload    | true    |         | audio file |                                                  |
| 目标角色id    | target_id | true    |         | int        |                                                  |
| 音频格式      | format    | true    |         | str        | wav,ogg,silk                                     |
| 语音长度/语速 | length    | true    |         | float      | 调节语音长度，相当于调节语速，该数值越大语速越慢 |
| 噪声          | noise     | true    |         | float      |                                                  |
| 噪声偏差      | noisew    | true    |         | float      |                                                  |

# communication

Learning and communication,now there is only Chinese [QQ group](https://qm.qq.com/cgi-bin/qm/qr?k=-1GknIe4uXrkmbDKBGKa1aAUteq40qs_&jump_from=webapi&authKey=x5YYt6Dggs1ZqWxvZqvj3fV8VUnxRyXm5S5Kzntc78+Nv3iXOIawplGip9LWuNR/)

# Acknowledgements

- vits:https://github.com/jaywalnut310/vits
- MoeGoe:https://github.com/CjangCjengh/MoeGoe

# 
