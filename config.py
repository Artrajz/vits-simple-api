import os
import sys

JSON_AS_ASCII = False
MAX_CONTENT_LENGTH = 5242880

# 端口
PORT = 23456
# 项目的绝对路径
ABS_PATH = os.path.join(os.path.dirname(os.path.realpath(sys.argv[0])))
# 上传文件的临时路径，非必要不要动
UPLOAD_FOLDER = ABS_PATH + "/upload"
# 音频转换的临时缓存路径，非必要不要动
CACHE_PATH = ABS_PATH + "/cache"

'''
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
'''
# 模型加载列表
MODEL_LIST = [

]
