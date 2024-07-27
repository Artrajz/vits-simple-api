from config import config
from manager.ModelManager import ModelManager
from manager.TTSManager import TTSManager

model_manager = ModelManager(config.system.device)
tts_manager = TTSManager(model_manager)

model_manager.attach(tts_manager)

model_manager.model_init()
