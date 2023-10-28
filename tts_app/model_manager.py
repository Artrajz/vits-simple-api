from ModelManager import ModelManager
from TTSManager import TTSManager
from config import MODEL_LIST

model_manager = ModelManager()
tts_manager = TTSManager(model_manager)

model_manager.attach(tts_manager)

model_manager.model_init(MODEL_LIST)
