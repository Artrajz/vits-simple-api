# MoeGoe-Simple-API



MoeGoe-Simple-API 是一个易部署的api，可以通过api的方式调用语音合成，可用于聊天机器人等，目前支持的功能有语音合成和语音转换。

- 可导入模型
- 支持加载多模型，可以将多个模型合并为一个新的id对应角色模型的映射表
<details><summary>点击预览返回的映射表</summary><pre><code>
[
	{
		"0": "綾地寧々"
	},
	{
		"1": "在原七海"
	},
	{
		"2": "小茸"
	},
	{
		"3": "唐乐吟"
	},
	{
		"...": "..."
	},
	{
		"196": "ルイズ"
	},
	{
		"197": "ティファニア"
	}
]
</code></pre></details>

# 如何使用

## Docker部署

### docker镜像拉取脚本

```
bash -c "$(wget -O- https://gist.githubusercontent.com/Artrajz/b2c02499f91c3a51b8b48f1a3c9a7ead/raw/e3033f1b222868b4b0f1b522e52e18217460ff91/moegoe-simple-api-installer-latest.sh)"
```

镜像大小为5g，所以拉取会比较慢，拉取成功后由于没有导入vits模型所以无法使用，需要按以下步骤导入模型

### 下载VITS模型

VITS模型放入`/usr/local/moegoe-simple-api/Model`文件夹中，模型文件夹中要有.pth和config.json文件

<details><summary>点击查看模型目录结构</summary><pre><code>
├─Model
│  ├─g
│  │      config.json
│  │      G_953000.pth
│  │
│  ├─Nene_Nanami_Rong_Tang
│  │      1374_epochs.pth
│  │      config.json
│  │
│  └─Zero_no_tsukaima
│          1158_epochs.pth
│          config.json
</code></pre></details>

### 修改模型路径

在 `/usr/local/moegoe-simple-api/config.py` 中修改模型路径

<details><summary>点击查看config.py模型路径填写示例</summary><pre><code>
vits模型路径填写方法，MODEL_LIST中的每一行是
[ABS_PATH+"/Model/{模型文件夹}/{.pth模型}", ABS_PATH+"/Model/{模型文件夹}/config.json"],
也可以写相对路径或绝对路径，由于windows和linux路径写法不同，用上面的写法或绝对路径最稳妥
示例：
MODEL_LIST = [
    [ABS_PATH+"/Model/Nene_Nanami_Rong_Tang/1374_epochs.pth", ABS_PATH+"/Model/Nene_Nanami_Rong_Tang/config.json"],
    [ABS_PATH+"/Model/Zero_no_tsukaima/1158_epochs.pth", ABS_PATH+"/Model/Zero_no_tsukaima/config.json"],
    [ABS_PATH+"/Model/g/G_953000.pth", ABS_PATH+"/Model/g/config.json"],
]
</code></pre></details>

### 开始使用！

终端输入`docker compose up -d`

或再次执行拉取脚本

### 镜像更新方法

重新执行docker镜像拉取脚本即可


## 直接部署

1. 下载VITS模型并放入`Model`文件夹中
2. 在 `config.py` 中修改模型路径
3. 安装python依赖（建议用conda虚拟环境） `pip install -r requirements.txt`
4. 开始使用！`python app.py`

## 参数

### 语音合成voice

| Name     | Parameter | Is must | Default | Value        | Instruction                               |
| -------- | --------- | ------- | ------- | ------------ | ----------------------------------------- |
| 合成文本 | text      | true    |         | text         |                                           |
| 角色id   | id        | false   | 0       | (number)     |                                           |
| 音频格式 | format    | false   | wav     | wav,ogg,silk | silk支持tx系语音                          |
| 文本语言 | lang      | false   | mix     | zh,ja,mix    | 当lang=mix时，文本应该用[ZH] 或 [JA] 包裹 |
| 语速     | speed     | false   | 1       | (number)     | 语速调节                                  |

### 语音转换voice conversion

| Name       | Parameter   | Is must | Default | Value      | Instruction            |
| ---------- | ----------- | ------- | ------- | ---------- | ---------------------- |
| 上传音频   | upload      | true    |         | audio file | 只支持wav和ogg         |
| 源角色id   | original_id | true    |         | (number)   | 上传文件所使用的角色id |
| 目标角色id | target_id   | true    |         | (number)   | 要转换的目标角色id     |

## GET

- GET/POST http://127.0.0.1:23456/voice/speakers

  返回id对应角色的映射表（json格式）

- GET http://127.0.0.1/voice?text=[JA]text[JA][ZH]text[ZH]&id=0&format=wav&lang=mix

  返回wav音频文件 

- GET http://127.0.0.1/voice?text=[JA]text[JA][ZH]text[ZH]&id=0&format=ogg&lang=mix

  返回ogg音频文件

- GET http://127.0.0.1/voice?text=text&lang=zh

  设定语言为zh，则文本无需[ZH]包裹

- GET http://127.0.0.1/voice?text=text&lang=ja

  设定语言为ja，则文本无需[JA]包裹

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

def voice_speakers():
    url = "http://127.0.0.1:23456/voice/speakers"

    res = requests.post(url=url)
    json = res.json()
    for i in json:
        print(i)
    
def voice():
    post_json = json.dumps({
        "text":"需要合成的文字",
        "id":172,
        "format":"wav",
        "lang":"zh"
        })
    headers={'content-type':'application/json'}
    url = "http://127.0.0.1:23456/voice"

    res = requests.post(url=url,data=post_json,headers=headers)
    fname = re.findall("filename=(.+)", res.headers["Content-Disposition"])[0]
    path = f"{abs_path}/{fname}"
    with open(path, "wb") as f:
        f.write(res.content)

def voice_conversion(upload_name):
    upload_path = f'{abs_path}/{upload_name}'
    upload_type = f'audio/{upload_name.split(".")[1]}' #wav,ogg
    
    fields = {
        "upload": (upload_name, open(upload_path,'rb'),upload_type),
        "original_id": "172",
        "target_id": "122",
    }
    boundary = '----VoiceConversionFormBoundary' \
               + ''.join(random.sample(string.ascii_letters + string.digits, 16))
    m = MultipartEncoder(fields=fields, boundary=boundary)
    
    headers = {"Content-Type": m.content_type}
    url = "http://127.0.0.1:23456/voice/conversion"

    res = requests.post(url=url,data=m,headers=headers)
    fname = re.findall("filename=(.+)", res.headers["Content-Disposition"])[0]
    path = f"{abs_path}/{fname}"
    with open(path, "wb") as f:
        f.write(res.content)
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
