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

- [x] VITS语音合成，语音转换
- [x] HuBert-soft VITS模型
- [x] W2V2 VITS / [emotional-vits](https://github.com/innnky/emotional-vits)维度情感模型
- [x] [vits_chinese](https://github.com/PlayVoice/vits_chinese)
- [x] [Bert-VITS2](https://github.com/Stardust-minus/Bert-VITS2)
- [x] 加载多模型
- [x] 自动识别语言并处理，根据模型的cleaner设置语言类型识别的范围，支持自定义语言类型范围
- [x] 自定义默认参数
- [x] 长文本批处理
- [x] GPU加速推理
- [x] SSML语音合成标记语言（完善中...）


## 在线demo

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/Artrajz/vits-simple-api) 感谢hugging face喵

注意不同的id支持的语言可能有所不同。[speakers](https://artrajz-vits-simple-api.hf.space/voice/speakers)


- `https://artrajz-vits-simple-api.hf.space/voice/vits?text=你好,こんにちは&id=164`
- `https://artrajz-vits-simple-api.hf.space/voice/vits?text=我觉得1%2B1≠3&id=164&lang=zh`（get中一些字符需要转义不然会被过滤掉）
- `https://artrajz-vits-simple-api.hf.space/voice/vits?text=Difficult the first time, easy the second.&id=4`
- 激动：`https://artrajz-vits-simple-api.hf.space/voice/w2v2-vits?text=こんにちは&id=3&emotion=111`
- 小声：`https://artrajz-vits-simple-api.hf.space/voice/w2v2-vits?text=こんにちは&id=3&emotion=2077`

https://user-images.githubusercontent.com/73542220/237995061-c1f25b4e-dd86-438a-9363-4bb1fe65b425.mov

# 部署

有两种部署方式可供选择。不论你选择哪一种，完成部署后都需要导入模型才能使用。

## Docker部署（Linux推荐）

### 步骤1: 镜像拉取

运行以下命令以拉取 Docker 镜像，根据脚本中的提示选择需要下载的必要文件和拉取镜像：

```bash
bash -c "$(wget -O- https://raw.githubusercontent.com/Artrajz/vits-simple-api/main/vits-simple-api-installer-latest.sh)"
```

项目配置文件以及模型文件夹的默认路径为`/usr/local/vits-simple-api/`

### 步骤2: 启动

运行以下命令启动容器：

```bash
docker-compose up -d
```

### 镜像更新

运行以下命令更新镜像：

```bash
docker-compose pull
```

重新启动容器：

```bash
docker-compose up -d
```

## 虚拟环境部署

### 步骤1: 克隆项目

使用以下命令克隆项目仓库：

```bash
git clone https://github.com/Artrajz/vits-simple-api.git
```

### 步骤2: 下载 Python 依赖

推荐使用 Python 虚拟环境。运行以下命令安装项目所需的 Python 依赖：

```bash
pip install -r requirements.txt
```

### 步骤3: 启动

运行以下命令启动程序：

```bash
python app.py
```

## Windows快速部署包

### 步骤1：下载并解压部署包

进入[releases页面](https://github.com/Artrajz/vits-simple-api/releases)下载并解压最新的部署包

### 步骤2：启动

运行start.bat启动程序

## 模型加载

### 步骤1: 下载 VITS 模型

将 VITS 模型文件下载并放入 `Model` 目录。

### 步骤2: 加载模型

如果是首次启动，在 `config.py` 文件中修改默认模型路径的配置。（非必须）

首次启动之后会生成一个config.yml配置文件，可以修改配置文件中的model_list或者在浏览器中进入管理员后台进行修改.

路径可填绝对路径或相对路径，相对路径则是从项目根目录中的Model文件夹开头。

比如Model文件夹中如下文件有

```
├─model1
│  │─G_1000.pth
│  └─config.json
└─model2
   │─G_1000.pth
   └─config.json
```

有多种可选的填法，按个人喜好选择

填法1

```yaml
'model_config':
  'model_list': 
  - - model1/G_1000.pth
    - model1/config.json
  - - model2/G_1000.pth
    - model2/config.json
```

填法2

```yaml
'model_config':
  'model_list': 
  - [model1/G_1000.pth, model1/config.json]
  - [model2/G_1000.pth, model2/config.json]
```

填法3

```yaml
'model_config':
  'model_list': [
    [model1/G_1000.pth, model1/config.json],
    [model2/G_1000.pth, model2/config.json],
  ]
```

# GPU 加速

## windows

### 安装CUDA

查看显卡最高支持CUDA的版本

```
nvidia-smi
```

以CUDA11.7为例，[官网](https://developer.nvidia.com/cuda-11-7-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exe_local)

### 安装GPU版pytorch

CUDA11.7对应的pytorch是用这个命令安装，推荐使用1.13.1+cu117，其他版本可能存在内存不稳定的问题。

```
pip install torch==1.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
```

## Linux

安装过程类似，可以查阅网上的安装资料。也可以直接使用docker部署脚本中的gpu版本。

# 功能选项说明

## 关闭管理员后台

由于管理员后台可以对模型进行加载和卸载操作，虽然有登录验证的保障，为了绝对安全，当对公网开放时，可以在`config.yml`中关闭管理员后台：

```yaml
'IS_ADMIN_ENABLED': !!bool 'false'
```

## Bert-VITS2配置使用语言/Bert模型

在Bert-VITS2 v2.0以后，一个模型需要加载三个不同语言的Bert模型。如果只需要使用其中一或两种语言，可以在模型的config.json的data中，添加`lang`参数，值为`['zh']`，表示该模型只使用中文，同时也只会加载中文的Bert模型。值为`['zh','ja']`表示只使用中日双语，同时也只会加载中文和日文的Bert模型。以此类推。

示例：

```json
"data": {
  "lang": ["zh","ja"],
  "training_files": "filelists/train.list",
  "validation_files": "filelists/val.list",
  "max_wav_value": 32768.0,
  ...
```

# 常见问题

## fasttext依赖安装问题

windows下可能安装不了fasttext,可以用以下命令安装，附[wheels下载地址](https://www.lfd.uci.edu/~gohlke/pythonlibs/#fasttext)

```
# python3.10 win_amd64
pip install https://github.com/Artrajz/archived/raw/main/fasttext/fasttext-0.9.2-cp310-cp310-win_amd64.whl
```

或者

```
pip install fasttext -i https://pypi.artrajz.cn/simple
```

## pyopenjtalk依赖安装问题

由于pypi.org没有pyopenjtalk的whl文件，通常需要从源代码来安装，这一过程对于一些人来说可能比较麻烦，所以你也可以使用我构建的whl来安装。

```
pip install pyopenjtalk -i https://pypi.artrajz.cn/simple
```

## Bert-VITS2版本兼容

修改Bert-VITS2模型的config.json，加入版本号参数`"version": "x.x.x"`，比如模型版本为1.0.1时，配置文件应该写成：

```
{
  "version": "1.0.1",
  "train": {
    "log_interval": 10,
    "eval_interval": 100,
    "seed": 52,
    ...
```

# 管理员后台

默认为http://127.0.0.1:23456/admin

初始账号密码在初次启动后，在config.yml最下方可找到。

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

- 见`api_test.py`



## API KEY

在config.py中设置`API_KEY_ENABLED = True`以启用，api key填写：`API_KEY = "api-key"`。

启用后，GET请求中使用需要增加参数api_key，POST请求中使用需要在header中添加参数`X-API-KEY`。

# Parameter

## VITS语音合成

| Name          | Parameter    | Is must | Default              | Type  | Instruction                                                  |
| ------------- | ------------ | ------- | -------------------- | ----- | ------------------------------------------------------------ |
| 合成文本      | text         | true    |                      | str   | 需要合成语音的文本。                                         |
| 角色id        | id           | false   | 从`config.yml`中获取 | int   | 即说话人id。                                                 |
| 音频格式      | format       | false   | 从`config.yml`中获取 | str   | 支持wav,ogg,silk,mp3,flac                                    |
| 文本语言      | lang         | false   | 从`config.yml`中获取 | str   | auto为自动识别语言模式，也是默认模式。lang=mix时，文本应该用[ZH] 或 [JA] 包裹。方言无法自动识别。 |
| 语音长度/语速 | length       | false   | 从`config.yml`中获取 | float | 调节语音长度，相当于调节语速，该数值越大语速越慢。           |
| 噪声          | noise        | false   | 从`config.yml`中获取 | float | 样本噪声，控制合成的随机性。                                 |
| sdp噪声       | noisew       | false   | 从`config.yml`中获取 | float | 随机时长预测器噪声，控制音素发音长度。                       |
| 分段阈值      | segment_size | false   | 从`config.yml`中获取 | int   | 按标点符号分段，加起来大于segment_size时为一段文本。segment_size<=0表示不分段。 |
| 流式响应      | streaming    | false   | false                | bool  | 流式合成语音，更快的首包响应。                               |

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

## W2V2-VITS

| Name          | Parameter    | Is must | Default              | Type  | Instruction                                                  |
| ------------- | ------------ | ------- | -------------------- | ----- | ------------------------------------------------------------ |
| 合成文本      | text         | true    |                      | str   | 需要合成语音的文本。                                         |
| 角色id        | id           | false   | 从`config.yml`中获取 | int   | 即说话人id。                                                 |
| 音频格式      | format       | false   | 从`config.yml`中获取 | str   | 支持wav,ogg,silk,mp3,flac                                    |
| 文本语言      | lang         | false   | 从`config.yml`中获取 | str   | auto为自动识别语言模式，也是默认模式。lang=mix时，文本应该用[ZH] 或 [JA] 包裹。方言无法自动识别。 |
| 语音长度/语速 | length       | false   | 从`config.yml`中获取 | float | 调节语音长度，相当于调节语速，该数值越大语速越慢             |
| 噪声          | noise        | false   | 从`config.yml`中获取 | float | 样本噪声，控制合成的随机性。                                 |
| sdp噪声       | noisew       | false   | 从`config.yml`中获取 | float | 随机时长预测器噪声，控制音素发音长度。                       |
| 分段阈值      | segment_size | false   | 从`config.yml`中获取 | int   | 按标点符号分段，加起来大于segment_size时为一段文本。segment_size<=0表示不分段。 |
| 维度情感      | emotion      | false   | 0                    | int   | 范围取决于npy情感参考文件，如[innnky](https://huggingface.co/spaces/innnky/nene-emotion/tree/main)的all_emotions.npy模型范围是0-5457 |

## Dimensional emotion

| Name     | Parameter | Is must | Default | Type | Instruction                   |
| -------- | --------- | ------- | ------- | ---- | ----------------------------- |
| 上传音频 | upload    | true    |         | file | 返回存储维度情感向量的npy文件 |

## Bert-VITS2语音合成

| Name          | Parameter       | Is must | Default              | Type  | Instruction                                                  |
| ------------- | --------------- | ------- | -------------------- | ----- | ------------------------------------------------------------ |
| 合成文本      | text            | true    |                      | str   | 需要合成语音的文本。                                         |
| 角色id        | id              | false   | 从`config.yml`中获取 | int   | 即说话人id。                                                 |
| 音频格式      | format          | false   | 从`config.yml`中获取 | str   | 支持wav,ogg,silk,mp3,flac                                    |
| 文本语言      | lang            | false   | 从`config.yml`中获取 | str   | auto为自动识别语言模式，也是默认模式，但目前只支持识别整段文本的语言，无法细分到每个句子。其余可选语言zh和ja。 |
| 语音长度/语速 | length          | false   | 从`config.yml`中获取 | float | 调节语音长度，相当于调节语速，该数值越大语速越慢。           |
| 噪声          | noise           | false   | 从`config.yml`中获取 | float | 样本噪声，控制合成的随机性。                                 |
| sdp噪声       | noisew          | false   | 从`config.yml`中获取 | float | 随机时长预测器噪声，控制音素发音长度。                       |
| 分段阈值      | segment_size    | false   | 从`config.yml`中获取 | int   | 按标点符号分段，加起来大于segment_size时为一段文本。segment_size<=0表示不分段。 |
| SDP/DP混合比  | sdp_ratio       | false   | 从`config.yml`中获取 | int   | SDP在合成时的占比，理论上此比率越高，合成的语音语调方差越大。 |
| 情感控制      | emotion         | false   | None                 |       | Bert-VITS2 v2.1可用，范围为0-9                               |
| 情感参考音频  | reference_audio | false   | None                 |       | Bert-VITS2 v2.1可用                                          |

## SSML语音合成标记语言
目前支持的元素与属性

`speak`元素

| Attribute    | Description                                                  | Is must |
| ------------ | ------------------------------------------------------------ | ------- |
| id           | 默认值从`config.yml`中读取                                   | false   |
| lang         | 默认值从`config.yml`中读取                                   | false   |
| length       | 默认值从`config.yml`中读取                                   | false   |
| noise        | 默认值从`config.yml`中读取                                   | false   |
| noisew       | 默认值从`config.yml`中读取                                   | false   |
| segment_size | 按标点符号分段，加起来大于segment_size时为一段文本。segment_size<=0表示不分段，这里默认为0。 | false   |
| model_type   | 默认为VITS，可选W2V2-VITS，BERT-VITS2                        | false   |
| emotion      | 只有用W2V2-VITS时`emotion`才会生效，范围取决于npy情感参考文件 | false   |
| sdp_ratio    | 只有用BERT-VITS2时`sdp_ratio`才会生效                        | false   |

`voice`元素

优先级大于`speak`

| Attribute    | Description                                                  | Is must |
| ------------ | ------------------------------------------------------------ | ------- |
| id           | 默认值从`config.yml`中读取                                   | false   |
| lang         | 默认值从`config.yml`中读取                                   | false   |
| length       | 默认值从`config.yml`中读取                                   | false   |
| noise        | 默认值从`config.yml`中读取                                   | false   |
| noisew       | 默认值从`config.yml`中读取                                   | false   |
| segment_size | 按标点符号分段，加起来大于segment_size时为一段文本。segment_size<=0表示不分段，这里默认为0。 | false   |
| model_type   | 默认为VITS，可选W2V2-VITS，BERT-VITS2                        | false   |
| emotion      | 只有用W2V2-VITS时`emotion`才会生效，范围取决于npy情感参考文件 | false   |
| sdp_ratio    | 只有用BERT-VITS2时`sdp_ratio`才会生效                        | false   |

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

见`api_test.py`

# 交流平台

现在只有 [Q群](https://qm.qq.com/cgi-bin/qm/qr?k=-1GknIe4uXrkmbDKBGKa1aAUteq40qs_&jump_from=webapi&authKey=x5YYt6Dggs1ZqWxvZqvj3fV8VUnxRyXm5S5Kzntc78+Nv3iXOIawplGip9LWuNR/)

# 鸣谢

- vits:https://github.com/jaywalnut310/vits
- MoeGoe:https://github.com/CjangCjengh/MoeGoe
- emotional-vits:https://github.com/innnky/emotional-vits
- vits-uma-genshin-honkai:https://huggingface.co/spaces/zomehwh/vits-uma-genshin-honkai
- vits_chinese:https://github.com/PlayVoice/vits_chinese
- Bert_VITS2:https://github.com/fishaudio/Bert-VITS2

# 感谢所有的贡献者

<a href="https://github.com/artrajz/vits-simple-ap/graphs/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=artrajz/vits-simple-api"/></a>
