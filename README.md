# MoeGoe-Simple-API



MoeGoe-Simple-API 是一个易部署的api，

- 可导入模型
- 支持多模型，可以将多个模型合并为一个新的id对应角色模型的映射表

```json
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
```

# 如何使用

1. 下载VITS模型并放入*Model*文件夹中
2. 在app.py中修改模型路径
3. 安装python依赖（建议用conda虚拟环境） `pip install -r requirements.txt`
4. 开始使用！`python app.py`

## 参数

| Name         | Parameter | Is must | Default | Value        | Instruction                               |
| ------------ | --------- | ------- | ------- | ------------ | ----------------------------------------- |
| text         | text      | true    |         | text         |                                           |
| speaker id   | id        | false   | 0       | (number)     |                                           |
| audio format | format    | false   | wav     | wav,ogg,silk | silk支持tx系语音                          |
| language     | lang      | false   | mix     | zh,ja,mix    | 当lang=mix时，文本应该用[ZH] 或 [JA] 包裹 |

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
import requests
import json

post_json = json.dumps({
    "text":"text here",
    "id":1,
    "format":"wav",
    "lang":"zh"
    })
headers={'content-type':'application/json'}
url = "http://127.0.0.1:23456/voice"

res = requests.post(url=url,data=post_json,headers=headers)

with open("audio.wav", "wb") as f:
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
