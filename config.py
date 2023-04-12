import os
import sys

JSON_AS_ASCII = False
MAX_CONTENT_LENGTH = 5242880

# port
PORT = 23456
# absolute path
ABS_PATH = os.path.join(os.path.dirname(os.path.realpath(sys.argv[0])))
# upload path
UPLOAD_FOLDER = ABS_PATH + "/upload"
# cahce path
CACHE_PATH = ABS_PATH + "/cache"
# zh ja ko en
LANGUAGE_AUTOMATIC_DETECT = ["zh","ja"]

'''
For each model, the filling method is as follows|模型列表中每个模型的填写方法如下
example|示例:
MODEL_LIST = [
    #VITS
    [ABS_PATH+"/Model/Nene_Nanami_Rong_Tang/1374_epochs.pth", ABS_PATH+"/Model/Nene_Nanami_Rong_Tang/config.json"],
    [ABS_PATH+"/Model/Zero_no_tsukaima/1158_epochs.pth", ABS_PATH+"/Model/Zero_no_tsukaima/config.json"],
    [ABS_PATH+"/Model/g/G_953000.pth", ABS_PATH+"/Model/g/config.json"],
    #HuBert-VITS
    [ABS_PATH+"/Model/louise/360_epochs.pth", ABS_PATH+"/Model/louise/config.json", ABS_PATH+"/Model/louise/hubert-soft-0d54a1f4.pt"],
]
'''
# load mutiple models
MODEL_LIST = [

]

"""
default params
以下选项是修改VITS GET方法 [不指定参数]时的默认值
"""
# 默认模式选项
DEFAULT_MODE = 0

if DEFAULT_MODE == 0:
    """默认选项0 自定义默认参数"""
    # GET 默认音色id
    ID = 0
    # GET 默认音频格式 可选wav,ogg,silk
    FORMAT = "wav"
    # GET 默认语言
    LANG = "AUTO"
    # GET 默认语音长度，相当于调节语速，该数值越大语速越慢
    LENGTH = 1
    # GET 默认噪声
    NOISE = 0.667
    # GET 默认噪声偏差
    NOISEW = 0.8
elif DEFAULT_MODE == 1:
    """进阶选项0 为每个音色自定义一套默认参数 有一定编程基础再用"""
    # vits有多少个音色
    nums_id = 4
    # GET 默认音色id
    ID = 0
    # GET 默认音频格式 可选wav,ogg,silk
    FORMAT = ["wav" for i in range(nums_id)]
    # GET 默认语言
    LANG = ["AUTO" for i in range(nums_id)]
    # GET 默认语音长度，相当于调节语速，该数值越大语速越慢
    LENGTH = [1 for i in range(nums_id)]
    # GET 默认噪声
    NOISE = [0.667 for i in range(nums_id)]
    # GET 默认噪声偏差
    NOISEW = [0.8 for i in range(nums_id)]
    """然后单独修改某个id的参数 自由发挥你的编程技术"""
    LANG[1] = "zh"
    NOISE[0], NOISE[1], NOISE[2] = 0.4, 0.4, 0.4
    NOISEW[0], NOISEW[1], NOISEW[2] = 0.4, 0.4, 0.4
