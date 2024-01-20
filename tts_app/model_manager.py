from manager.ModelManager import ModelManager
from manager.TTSManager import TTSManager

model_manager = ModelManager()
tts_manager = TTSManager(model_manager)

model_manager.attach(tts_manager)

model_manager.model_init()
