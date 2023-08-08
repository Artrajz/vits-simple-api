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

- [x] VITS语音合成
- [x] VITS语音转换
- [x] HuBert-soft VITS模型
- [x] W2V2 VITS / emotional-vits维度情感模型
- [x] 加载多模型
- [x] 自动识别语言并处理，根据模型的cleaner设置语言类型识别的范围，支持自定义语言类型范围
- [x] 自定义默认参数
- [x] 长文本批处理
- [x] GPU加速推理
- [x] SSML语音合成标记语言（完善中...）

<details><summary>Update Logs</summary><pre><code>
<h2>2023.6.5</h2>
<p>更换音频编码使用的库，增加flac格式，增加中文对读简单数学公式的支持</p>
<h2>2023.5.24</h2>
<p>添加dimensional_emotion api,从文件夹加载多个npy文件,Docker添加了Linux/ARM64和Linux/ARM64/v8平台</p>
<h2>2023.5.15</h2>
<p>增加english_cleaner，需要额外安装espeak才能使用</p>
<h2>2023.5.12</h2>
<p>增加ssml支持，但仍需完善。重构部分功能，hubert_vits中的speaker_id改为id</p>
<h2>2023.5.2</h2>
<p>增加w2v2-vits/emotional-vits模型支持，修改了speakers映射表并添加了对应模型支持的语言</p>
<h2>2023.4.23</h2>
<p>增加api key鉴权，默认禁用，需要在config.py中启用</p>
<h2>2023.4.17</h2>
<p>修改单语言的cleaner需要标注才会clean，增加GPU加速推理，但需要手动安装gpu推理环境</p>
<h2>2023.4.12</h2>
<p>项目由MoeGoe-Simple-API更名为vits-simple-api，支持长文本批处理，增加长文本分段阈值max</p>
<h2>2023.4.7</h2>
<p>增加配置文件可自定义默认参数，本次更新需要手动更新config.py，具体使用方法见config.py</p>
<h2>2023.4.6</h2>
<p>加入自动识别语种选项auto，lang参数默认修改为auto，自动识别仍有一定缺陷，请自行选择</p>
<p>统一POST请求类型为multipart/form-data</p>
</code></pre></details>



## demo

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/Artrajz/vits-simple-api)

注意不同的id支持的语言可能有所不同。[speakers](https://artrajz-vits-simple-api.hf.space/voice/speakers)


- `https://artrajz-vits-simple-api.hf.space/voice/vits?text=你好,こんにちは&id=164`
- `https://artrajz-vits-simple-api.hf.space/voice/vits?text=我觉得1%2B1≠3&id=164&lang=zh`（get中一些字符需要转义不然会被过滤掉）
- `https://artrajz-vits-simple-api.hf.space/voice/vits?text=Difficult the first time, easy the second.&id=4`
- 激动：`https://artrajz-vits-simple-api.hf.space/voice/w2v2-vits?text=こんにちは&id=3&emotion=111`
- 小声：`https://artrajz-vits-simple-api.hf.space/voice/w2v2-vits?text=こんにちは&id=3&emotion=2077`

https://user-images.githubusercontent.com/73542220/237995061-c1f25b4e-dd86-438a-9363-4bb1fe65b425.mov

# 部署

## Docker部署

### 镜像拉取脚本

```
bash -c "$(wget -O- https://raw.githubusercontent.com/Artrajz/vits-simple-api/main/vits-simple-api-installer-latest.sh)"
```

- 目前docker镜像支持的平台`linux/amd64,linux/arm64`
- 在拉取完成后，需要导入VITS模型才能使用，请根据以下步骤导入模型。

### 下载VITS模型

将模型放入`/usr/local/vits-simple-api/Model`

<details><summary>Folder structure</summary><pre><code>
│  hubert-soft-0d54a1f4.pt
│  model.onnx
│  model.yaml
├─g
│      config.json
│      G_953000.pth
│
├─louise
│      360_epochs.pth
│      config.json
│
├─Nene_Nanami_Rong_Tang
│      1374_epochs.pth
│      config.json
│
├─Zero_no_tsukaima
│       1158_epochs.pth
│       config.json
│
└─npy
       25ecb3f6-f968-11ed-b094-e0d4e84af078.npy
       all_emotions.npy
</code></pre></details>



### 修改模型路径

Modify in  `/usr/local/vits-simple-api/config.py` 

<details><summary>config.py</summary><pre><code>
# 在此填写模型路径
MODEL_LIST = [
    # VITS
    [ABS_PATH + "/Model/Nene_Nanami_Rong_Tang/1374_epochs.pth", ABS_PATH + "/Model/Nene_Nanami_Rong_Tang/config.json"],
    [ABS_PATH + "/Model/Zero_no_tsukaima/1158_epochs.pth", ABS_PATH + "/Model/Zero_no_tsukaima/config.json"],
    [ABS_PATH + "/Model/g/G_953000.pth", ABS_PATH + "/Model/g/config.json"],
    # HuBert-VITS (Need to configure HUBERT_SOFT_MODEL)
    [ABS_PATH + "/Model/louise/360_epochs.pth", ABS_PATH + "/Model/louise/config.json"],
    # W2V2-VITS (Need to configure DIMENSIONAL_EMOTION_NPY)
    [ABS_PATH + "/Model/w2v2-vits/1026_epochs.pth", ABS_PATH + "/Model/w2v2-vits/config.json"],
]
# hubert-vits: hubert soft 编码器
HUBERT_SOFT_MODEL = ABS_PATH + "/Model/hubert-soft-0d54a1f4.pt"
# w2v2-vits: Dimensional emotion npy file
# 加载单独的npy: ABS_PATH+"/all_emotions.npy
# 加载多个npy: [ABS_PATH + "/emotions1.npy", ABS_PATH + "/emotions2.npy"]
# 从文件夹里加载npy: ABS_PATH + "/Model/npy"
DIMENSIONAL_EMOTION_NPY = ABS_PATH + "/Model/npy"
# w2v2-vits: 需要在同一路径下有model.onnx和model.yaml
DIMENSIONAL_EMOTION_MODEL = ABS_PATH + "/Model/model.yaml"
</code></pre></details>



### 启动

`docker compose up -d`

或者重新执行拉取脚本

### 镜像更新

重新执行docker镜像拉取脚本即可

## 虚拟环境部署

### Clone

`git clone https://github.com/Artrajz/vits-simple-api.git`

###  下载python依赖

推荐使用python的虚拟环境，python版本 >= 3.9

`pip install -r requirements.txt`

windows下可能安装不了fasttext,可以用以下命令安装，附[wheels下载地址](https://www.lfd.uci.edu/~gohlke/pythonlibs/#fasttext)

```
#python3.10 win_amd64
pip install https://github.com/Artrajz/archived/raw/main/fasttext/fasttext-0.9.2-cp310-cp310-win_amd64.whl
#python3.9 win_amd64
pip install https://github.com/Artrajz/archived/raw/main/fasttext/fasttext-0.9.2-cp39-cp39-win_amd64.whl
```

### 下载VITS模型

将模型放入 `/path/to/vits-simple-api/Model`

<details><summary>文件夹结构</summary><pre><code>
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

### 修改模型路径

在 `/path/to/vits-simple-api/config.py` 修改

<details><summary>config.py</summary><pre><code>
# 在此填写模型路径
MODEL_LIST = [
    # VITS
    [ABS_PATH + "/Model/Nene_Nanami_Rong_Tang/1374_epochs.pth", ABS_PATH + "/Model/Nene_Nanami_Rong_Tang/config.json"],
    [ABS_PATH + "/Model/Zero_no_tsukaima/1158_epochs.pth", ABS_PATH + "/Model/Zero_no_tsukaima/config.json"],
    [ABS_PATH + "/Model/g/G_953000.pth", ABS_PATH + "/Model/g/config.json"],
    # HuBert-VITS (Need to configure HUBERT_SOFT_MODEL)
    [ABS_PATH + "/Model/louise/360_epochs.pth", ABS_PATH + "/Model/louise/config.json"],
    # W2V2-VITS (Need to configure DIMENSIONAL_EMOTION_NPY)
    [ABS_PATH + "/Model/w2v2-vits/1026_epochs.pth", ABS_PATH + "/Model/w2v2-vits/config.json"],
]
# hubert-vits: hubert soft 编码器
HUBERT_SOFT_MODEL = ABS_PATH + "/Model/hubert-soft-0d54a1f4.pt"
# w2v2-vits: Dimensional emotion npy file
# 加载单独的npy: ABS_PATH+"/all_emotions.npy
# 加载多个npy: [ABS_PATH + "/emotions1.npy", ABS_PATH + "/emotions2.npy"]
# 从文件夹里加载npy: ABS_PATH + "/Model/npy"
DIMENSIONAL_EMOTION_NPY = ABS_PATH + "/Model/npy"
# w2v2-vits: 需要在同一路径下有model.onnx和model.yaml
DIMENSIONAL_EMOTION_MODEL = ABS_PATH + "/Model/model.yaml"
</code></pre></details>



### 启动

`python app.py`

# GPU 加速

## windows

### 安装CUDA

查看显卡最高支持CUDA的版本

```
nvidia-smi
```

以CUDA11.7为例，[官网](https://developer.nvidia.com/cuda-11-7-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exe_local)

### 安装GPU版pytorch

CUDA11.7对应的pytorch是用这个命令安装

```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```

对应版本的命令可以在[官网](https://pytorch.org/get-started/locally/)找到

## Linux

安装过程类似，但我没有相应的环境所以没办法测试

# Openjtalk安装问题

如果你是arm64架构的平台，由于pypi官网上没有arm64对应的whl，可能安装会出现一些问题，你可以使用我构建的whl来安装

```
pip install openjtalk==0.3.0.dev2 --index-url https://pypi.artrajz.cn/simple
```

或者是自己手动构建一个whl，可以根据[教程](https://artrajz.cn/index.php/archives/167/)来构建

# API

## GET

#### speakers list 

- GET http://127.0.0.1:23456/voice/speakers

  返回id对应角色的映射表

#### voice vits

- GET http://127.0.0.1:23456/voice/vits?text=text

  其他参数不指定时均为默认值

- GET http://127.0.0.1:23456/voice/vits?text=[ZH]text[ZH][JA]text[JA]&lang=mix

  lang=mix时文本要标注

- GET http://127.0.0.1:23456/voice/vits?text=text&id=142&format=wav&lang=zh&length=1.4

  文本为text，角色id为142，音频格式为wav，文本语言为zh，语音长度为1.4，其余参数默认

#### check

- GET http://127.0.0.1:23456/voice/check?id=0&model=vits

## POST

- python

```python
import re
import requests
import os
import random
import string
from requests_toolbelt.multipart.encoder import MultipartEncoder

abs_path = os.path.dirname(__file__)
base = "http://127.0.0.1:23456"


# 映射表
def voice_speakers():
    url = f"{base}/voice/speakers"

    res = requests.post(url=url)
    json = res.json()
    for i in json:
        print(i)
        for j in json[i]:
            print(j)
    return json


# 语音合成 voice vits
def voice_vits(text, id=0, format="wav", lang="auto", length=1, noise=0.667, noisew=0.8, max=50):
    fields = {
        "text": text,
        "id": str(id),
        "format": format,
        "lang": lang,
        "length": str(length),
        "noise": str(noise),
        "noisew": str(noisew),
        "max": str(max)
    }
    boundary = '----VoiceConversionFormBoundary' + ''.join(random.sample(string.ascii_letters + string.digits, 16))

    m = MultipartEncoder(fields=fields, boundary=boundary)
    headers = {"Content-Type": m.content_type}
    url = f"{base}/voice"

    res = requests.post(url=url, data=m, headers=headers)
    fname = re.findall("filename=(.+)", res.headers["Content-Disposition"])[0]
    path = f"{abs_path}/{fname}"

    with open(path, "wb") as f:
        f.write(res.content)
    print(path)
    return path


# 语音转换 hubert-vits
def voice_hubert_vits(upload_path, id, format="wav", length=1, noise=0.667, noisew=0.8):
    upload_name = os.path.basename(upload_path)
    upload_type = f'audio/{upload_name.split(".")[1]}'  # wav,ogg

    with open(upload_path, 'rb') as upload_file:
        fields = {
            "upload": (upload_name, upload_file, upload_type),
            "id": str(id),
            "format": format,
            "length": str(length),
            "noise": str(noise),
            "noisew": str(noisew),
        }
        boundary = '----VoiceConversionFormBoundary' + ''.join(random.sample(string.ascii_letters + string.digits, 16))

        m = MultipartEncoder(fields=fields, boundary=boundary)
        headers = {"Content-Type": m.content_type}
        url = f"{base}/voice/hubert-vits"

        res = requests.post(url=url, data=m, headers=headers)
    fname = re.findall("filename=(.+)", res.headers["Content-Disposition"])[0]
    path = f"{abs_path}/{fname}"

    with open(path, "wb") as f:
        f.write(res.content)
    print(path)
    return path


# 维度情感模型 w2v2-vits
def voice_w2v2_vits(text, id=0, format="wav", lang="auto", length=1, noise=0.667, noisew=0.8, max=50, emotion=0):
    fields = {
        "text": text,
        "id": str(id),
        "format": format,
        "lang": lang,
        "length": str(length),
        "noise": str(noise),
        "noisew": str(noisew),
        "max": str(max),
        "emotion": str(emotion)
    }
    boundary = '----VoiceConversionFormBoundary' + ''.join(random.sample(string.ascii_letters + string.digits, 16))

    m = MultipartEncoder(fields=fields, boundary=boundary)
    headers = {"Content-Type": m.content_type}
    url = f"{base}/voice/w2v2-vits"

    res = requests.post(url=url, data=m, headers=headers)
    fname = re.findall("filename=(.+)", res.headers["Content-Disposition"])[0]
    path = f"{abs_path}/{fname}"

    with open(path, "wb") as f:
        f.write(res.content)
    print(path)
    return path


# 语音转换 同VITS模型内角色之间的音色转换
def voice_conversion(upload_path, original_id, target_id):
    upload_name = os.path.basename(upload_path)
    upload_type = f'audio/{upload_name.split(".")[1]}'  # wav,ogg

    with open(upload_path, 'rb') as upload_file:
        fields = {
            "upload": (upload_name, upload_file, upload_type),
            "original_id": str(original_id),
            "target_id": str(target_id),
        }
        boundary = '----VoiceConversionFormBoundary' + ''.join(random.sample(string.ascii_letters + string.digits, 16))
        m = MultipartEncoder(fields=fields, boundary=boundary)

        headers = {"Content-Type": m.content_type}
        url = f"{base}/voice/conversion"

        res = requests.post(url=url, data=m, headers=headers)

    fname = re.findall("filename=(.+)", res.headers["Content-Disposition"])[0]
    path = f"{abs_path}/{fname}"

    with open(path, "wb") as f:
        f.write(res.content)
    print(path)
    return path


def voice_ssml(ssml):
    fields = {
        "ssml": ssml,
    }
    boundary = '----VoiceConversionFormBoundary' + ''.join(random.sample(string.ascii_letters + string.digits, 16))

    m = MultipartEncoder(fields=fields, boundary=boundary)
    headers = {"Content-Type": m.content_type}
    url = f"{base}/voice/ssml"

    res = requests.post(url=url, data=m, headers=headers)
    fname = re.findall("filename=(.+)", res.headers["Content-Disposition"])[0]
    path = f"{abs_path}/{fname}"

    with open(path, "wb") as f:
        f.write(res.content)
    print(path)
    return path

def voice_dimensional_emotion(upload_path):
    upload_name = os.path.basename(upload_path)
    upload_type = f'audio/{upload_name.split(".")[1]}'  # wav,ogg

    with open(upload_path, 'rb') as upload_file:
        fields = {
            "upload": (upload_name, upload_file, upload_type),
        }
        boundary = '----VoiceConversionFormBoundary' + ''.join(random.sample(string.ascii_letters + string.digits, 16))

        m = MultipartEncoder(fields=fields, boundary=boundary)
        headers = {"Content-Type": m.content_type}
        url = f"{base}/voice/dimension-emotion"

        res = requests.post(url=url, data=m, headers=headers)
    fname = re.findall("filename=(.+)", res.headers["Content-Disposition"])[0]
    path = f"{abs_path}/{fname}"

    with open(path, "wb") as f:
        f.write(res.content)
    print(path)
    return path
```

## API KEY

在config.py中设置`API_KEY_ENABLED = True`以启用，api key填写：`API_KEY = "api-key"`。

启用后，GET请求中使用需要增加参数api_key，POST请求中使用需要在header中添加参数`X-API-KEY`。

# Parameter

## VITS语音合成

| Name          | Parameter | Is must | Default | Type  | Instruction                                                  |
| ------------- | --------- | ------- | ------- | ----- | ------------------------------------------------------------ |
| 合成文本      | text      | true    |         | str   | 需要合成语音的文本。                                         |
| 角色id        | id        | false   | 0       | int   | 即说话人id。                                                 |
| 音频格式      | format    | false   | wav     | str   | 支持wav,ogg,silk,mp3,flac                                    |
| 文本语言      | lang      | false   | auto    | str   | auto为自动识别语言模式，也是默认模式。lang=mix时，文本应该用[ZH] 或 [JA] 包裹。方言无法自动识别。 |
| 语音长度/语速 | length    | false   | 1.0     | float | 调节语音长度，相当于调节语速，该数值越大语速越慢。           |
| 噪声          | noise     | false   | 0.33    | float | 样本噪声，控制合成的随机性。                                 |
| sdp噪声       | noisew    | false   | 0.4     | float | 随机时长预测器噪声，控制音素发音长度。                       |
| 分段阈值      | max       | false   | 50      | int   | 按标点符号分段，加起来大于max时为一段文本。max<=0表示不分段。 |
| 流式响应      | streaming | false   | false   | bool  | 流式合成语音，更快的首包响应。                               |

## VITS 语音转换

| Name       | Parameter   | Is must | Default | Type | Instruction            |
| ---------- | ----------- | ------- | ------- | ---- | ---------------------- |
| 上传音频   | upload      | true    |         | file | wav or ogg             |
| 源角色id   | original_id | true    |         | int  | 上传文件所使用的角色id |
| 目标角色id | target_id   | true    |         | int  | 要转换的目标角色id     |

## HuBert-VITS 语音转换

| Name          | Parameter | Is must | Default | Type  | Instruction                                      |
| ------------- | --------- | ------- | ------- | ----- | ------------------------------------------------ |
| 上传音频      | upload    | true    |         | file  | 需要转换说话人的音频文件。                       |
| 目标角色id    | id        | true    |         | int   | 目标说话人id。                                   |
| 音频格式      | format    | true    |         | str   | wav,ogg,silk                                     |
| 语音长度/语速 | length    | true    |         | float | 调节语音长度，相当于调节语速，该数值越大语速越慢 |
| 噪声          | noise     | true    |         | float | 样本噪声，控制合成的随机性。                     |
| sdp噪声       | noisew    | true    |         | float | 随机时长预测器噪声，控制音素发音长度。           |

## Dimensional emotion

| Name     | Parameter | Is must | Default | Type | Instruction                   |
| -------- | --------- | ------- | ------- | ---- | ----------------------------- |
| 上传音频 | upload    | true    |         | file | 返回存储维度情感向量的npy文件 |

## W2V2-VITS

| Name          | Parameter | Is must | Default | Type  | Instruction                                                  |
| ------------- | --------- | ------- | ------- | ----- | ------------------------------------------------------------ |
| 合成文本      | text      | true    |         | str   | 需要合成语音的文本。                                         |
| 角色id        | id        | false   | 0       | int   | 即说话人id。                                                 |
| 音频格式      | format    | false   | wav     | str   | 支持wav,ogg,silk,mp3,flac                                    |
| 文本语言      | lang      | false   | auto    | str   | auto为自动识别语言模式，也是默认模式。lang=mix时，文本应该用[ZH] 或 [JA] 包裹。方言无法自动识别。 |
| 语音长度/语速 | length    | false   | 1.0     | float | 调节语音长度，相当于调节语速，该数值越大语速越慢             |
| 噪声          | noise     | false   | 0.33    | float | 样本噪声，控制合成的随机性。                                 |
| sdp噪声       | noisew    | false   | 0.4     | float | 随机时长预测器噪声，控制音素发音长度。                       |
| 分段阈值      | max       | false   | 50      | int   | 按标点符号分段，加起来大于max时为一段文本。max<=0表示不分段。 |
| 维度情感      | emotion   | false   | 0       | int   | 范围取决于npy情感参考文件，如[innnky](https://huggingface.co/spaces/innnky/nene-emotion/tree/main)的all_emotions.npy模型范围是0-5457 |

## SSML语音合成标记语言
目前支持的元素与属性

`speak`元素

| Attribute | Description                                                  | Is must |
| --------- | ------------------------------------------------------------ | ------- |
| id        | 默认值从`config.py`中读取                                    | false   |
| lang      | 默认值从`config.py`中读取                                    | false   |
| length    | 默认值从`config.py`中读取                                    | false   |
| noise     | 默认值从`config.py`中读取                                    | false   |
| noisew    | 默认值从`config.py`中读取                                    | false   |
| max       | 按标点符号分段，加起来大于max时为一段文本。max<=0表示不分段，这里默认为0。 | false   |
| model     | 默认为vits，可选`w2v2-vits`，`emotion-vits`                  | false   |
| emotion   | 只有用`w2v2-vits`或`emotion-vits`时`emotion`才生效，范围取决于npy情感参考文件 | false   |

`voice`元素

优先级大于`speak`

| Attribute | Description                                                  | Is must |
| --------- | ------------------------------------------------------------ | ------- |
| id        | 默认值从`config.py`中读取                                    | false   |
| lang      | 默认值从`config.py`中读取                                    | false   |
| length    | 默认值从`config.py`中读取                                    | false   |
| noise     | 默认值从`config.py`中读取                                    | false   |
| noisew    | 默认值从`config.py`中读取                                    | false   |
| max       | 按标点符号分段，加起来大于max时为一段文本。max<=0表示不分段，这里默认为0。 | false   |
| model     | 默认为vits，可选`w2v2-vits`，`emotion-vits`                  | false   |
| emotion   | 只有用`w2v2-vits`或`emotion-vits`时`emotion`才会生效         | false   |

`break`元素

| Attribute | Description                                                  | Is must |
| --------- | ------------------------------------------------------------ | ------- |
| strength  | x-weak,weak,medium（默认值）,strong,x-strong                 | false   |
| time      | 暂停的绝对持续时间，以秒为单位（例如 `2s`）或以毫秒为单位（例如 `500ms`）。 有效值的范围为 0 到 5000 毫秒。 如果设置的值大于支持的最大值，则服务将使用 `5000ms`。 如果设置了 `time` 属性，则会忽略 `strength` 属性。 | false   |

| Strength | Relative Duration |
| :------- | :---------------- |
| x-weak   | 250 毫秒          |
| weak     | 500 毫秒          |
| Medium   | 750 毫秒          |
| Strong   | 1000 毫秒         |
| x-strong | 1250 毫秒         |

示例

```xml
<speak lang="zh" format="mp3" length="1.2">
    <voice id="92" >这几天心里颇不宁静。</voice>
    <voice id="125">今晚在院子里坐着乘凉，忽然想起日日走过的荷塘，在这满月的光里，总该另有一番样子吧。</voice>
    <voice id="142">月亮渐渐地升高了，墙外马路上孩子们的欢笑，已经听不见了；</voice>
    <voice id="98">妻在屋里拍着闰儿，迷迷糊糊地哼着眠歌。</voice>
    <voice id="120">我悄悄地披了大衫，带上门出去。</voice><break time="2s"/>
    <voice id="121">沿着荷塘，是一条曲折的小煤屑路。</voice>
    <voice id="122">这是一条幽僻的路；白天也少人走，夜晚更加寂寞。</voice>
    <voice id="123">荷塘四面，长着许多树，蓊蓊郁郁的。</voice>
    <voice id="124">路的一旁，是些杨柳，和一些不知道名字的树。</voice>
    <voice id="125">没有月光的晚上，这路上阴森森的，有些怕人。</voice>
    <voice id="126">今晚却很好，虽然月光也还是淡淡的。</voice><break time="2s"/>
    <voice id="127">路上只我一个人，背着手踱着。</voice>
    <voice id="128">这一片天地好像是我的；我也像超出了平常的自己，到了另一个世界里。</voice>
    <voice id="129">我爱热闹，也爱冷静；<break strength="x-weak"/>爱群居，也爱独处。</voice>
    <voice id="130">像今晚上，一个人在这苍茫的月下，什么都可以想，什么都可以不想，便觉是个自由的人。</voice>
    <voice id="131">白天里一定要做的事，一定要说的话，现在都可不理。</voice>
    <voice id="132">这是独处的妙处，我且受用这无边的荷香月色好了。</voice>
</speak>
```

# 交流平台

现在只有 [Q群](https://qm.qq.com/cgi-bin/qm/qr?k=-1GknIe4uXrkmbDKBGKa1aAUteq40qs_&jump_from=webapi&authKey=x5YYt6Dggs1ZqWxvZqvj3fV8VUnxRyXm5S5Kzntc78+Nv3iXOIawplGip9LWuNR/)

# 鸣谢

- vits:https://github.com/jaywalnut310/vits
- MoeGoe:https://github.com/CjangCjengh/MoeGoe
- emotional-vits:https://github.com/innnky/emotional-vits
- vits-uma-genshin-honkai:https://huggingface.co/spaces/zomehwh/vits-uma-genshin-honkai
- vits_chinese:https://github.com/PlayVoice/vits_chinese

