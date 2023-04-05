<div class="title" align=center>
    <h1>MoeGoe-Simple-API</h1>
	<div>MoeGoe-Simple-API 是一个易部署的api，可以通过api的方式调用语音合成，可用于聊天机器人等。</div>
    <br/>
    <br/>
    <p>
        <img src="https://img.shields.io/github/license/Artrajz/MoeGoe-Simple-API">
   		<img src="https://img.shields.io/badge/python-3.9%7C3.10-green">
    </p>
</div>
<hr style="height:1px;border:none;border-top:1px solid var(--color-border-muted)" />



#### 目前支持的功能

- vits语音合成
- hubert-vits语音转换
- 同vits模型内的语音转换
- 加载多模型，将多个模型合并为一个新的id对应角色模型的映射表（不同类别的模型是分开的列表）

<details><summary>点击预览返回的映射表</summary><pre><code>
{"HuBert-VITS":[{"0":"ルイズ"}],"VITS":[{"0":"綾地寧々"},{"1":"在原七海"},{"2":"小茸"},{"3":"唐乐吟"}],"W2V2-VITS":[]}
</code></pre></details>


#### 测试API

`https://api.artrajz.cn/py/voice?text=喂？听得到吗&id=142&lang=zh`

不保证服务器稳定性，请勿滥用

# 如何部署

## Docker部署

### docker镜像拉取脚本

```
bash -c "$(wget -O- https://raw.githubusercontent.com/Artrajz/MoeGoe-Simple-API/main/moegoe-simple-api-installer-latest.sh)"
```

- 镜像大小为5g，所以拉取会比较慢，解压后为8g，请准备足够的磁盘空间
- 拉取成功后由于没有导入vits模型所以无法使用，需要按以下步骤导入模型

### 下载VITS模型

VITS模型放入`/usr/local/moegoe-simple-api/Model`文件夹中，模型文件夹中要有.pth和config.json文件

<details><summary>点击查看Model文件夹结构</summary><pre><code>
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

在 `/usr/local/moegoe-simple-api/config.py` 中修改模型路径

<details><summary>点击查看config.py模型路径填写示例</summary><pre><code>
vits模型路径填写方法，MODEL_LIST中的每一行是
[ABS_PATH+"/Model/{模型文件夹}/{.pth模型}", ABS_PATH+"/Model/{模型文件夹}/config.json"],
也可以写相对路径或绝对路径，由于windows和linux路径写法不同，用上面的写法或绝对路径最稳妥
示例：
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

### 镜像更新方法

重新执行docker镜像拉取脚本即可

## 直接部署

python3.9以上版本

1. 下载VITS模型并放入`Model`文件夹中
2. 在 `config.py` 中修改模型路径
3. 安装python依赖（建议用conda虚拟环境） `pip install -r requirements.txt`
4. 开始使用！`python app.py`

# 参数

## 语音合成voice vits

| Name          | Parameter | Is must | Default | Value        | Instruction                                      |
| ------------- | --------- | ------- | ------- | ------------ | ------------------------------------------------ |
| 合成文本      | text      | true    |         | text         |                                                  |
| 角色id        | id        | false   | 0       | (int)        |                                                  |
| 音频格式      | format    | false   | wav     | wav,ogg,silk | silk支持tx系语音                                 |
| 文本语言      | lang      | false   | mix     | zh,ja,mix    | 当lang=mix时，文本应该用[ZH] 或 [JA] 包裹        |
| 语音长度/语速 | length    | false   | 1.0     | (float)      | 调节语音长度，相当于调节语速，该数值越大语速越慢 |
| 噪声          | noise     | false   | 0.667   | (float)      | 噪声微调，一般用默认值即可                       |
| 噪声偏差      | noisew    | false   | 0.8     | (float)      | 噪声偏差微调，一般用默认值即可                   |

## 语音转换voice conversion

| Name       | Parameter   | Is must | Default | Value      | Instruction            |
| ---------- | ----------- | ------- | ------- | ---------- | ---------------------- |
| 上传音频   | upload      | true    |         | audio file | 只支持wav和ogg         |
| 源角色id   | original_id | true    |         | (number)   | 上传文件所使用的角色id |
| 目标角色id | target_id   | true    |         | (number)   | 要转换的目标角色id     |

## 语音转换 HuBert-VITS

| Name          | Parameter | Is must | Default | Value        | Instruction                                      |
| ------------- | --------- | ------- | ------- | ------------ | ------------------------------------------------ |
| 上传音频      | upload    | true    |         | audio file   | 只支持wav和ogg                                   |
| 目标角色id    | id        | true    |         | (int)        |                                                  |
| 音频格式      | format    | true    |         | wav,ogg,silk | silk支持tx系语音                                 |
| 文本语言      | lang      | true    |         | zh,ja,mix    | 当lang=mix时，文本应该用[ZH] 或 [JA] 包裹        |
| 语音长度/语速 | length    | true    |         | (float)      | 调节语音长度，相当于调节语速，该数值越大语速越慢 |
| 噪声          | noise     | true    |         | (float)      | 噪声微调                                         |
| 噪声偏差      | noisew    | true    |         | (float)      | 噪声偏差微调                                     |

# 调用方法

## GET

#### 映射表

- GET/POST http://127.0.0.1:23456/voice/speakers

  返回id对应角色的映射表（json格式）

#### 语音合成voice vits

- GET http://127.0.0.1/voice?text=[JA]text[JA][ZH]text[ZH]&id=0&format=wav&lang=mix

  返回wav音频文件

- GET http://127.0.0.1/voice?text=[JA]text[JA][ZH]text[ZH]&id=0&format=ogg&lang=mix

  返回ogg音频文件

- GET http://127.0.0.1/voice?text=text&lang=zh

  设定语言为zh，则文本无需[ZH]包裹

- GET http://127.0.0.1/voice?text=text&lang=ja

  设定语言为ja，则文本无需[JA]包裹

- GET http://127.0.0.1/voice?text=text&id=142&format=wav&lang=zh&length=1.4

  文本为text，角色id为142，音频格式为wav，文本语言为zh，语音长度为1.4，其余参数默认

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
def voice_vits():
    post_json = json.dumps({
        "text":"你好，我是艾草",
        "id":"3",
        "format":"wav",
        "lang":"zh",
        "length":"1",
        "noise":"0.667",
        "noisew":"0.8",
        })
    headers={'content-type':'application/json'}
    url = f"{addr}/voice"

    res = requests.post(url=url,data=post_json,headers=headers)
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

# 可能遇到的问题

~~本人遇到过的问题~~

### 运行后服务器无响应

可能是内存不足，可以尝试减少模型加载数量。

### 模型推理时服务器无响应

可能是同时处理多个推理任务导致CPU堵塞，可以尝试在*voice.py*中取消以下**两行**代码的注释，意思是让pytorch只使用1个物理CPU核心，防止一个任务抢占过多CPU资源。

```python
import torch
torch.set_num_threads(1)
```

# 鸣谢

 该项目基于[CjangCjengh](https://github.com/CjangCjengh)的[MoeGoe](https://github.com/CjangCjengh/MoeGoe)
