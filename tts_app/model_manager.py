from ModelManager import ModelManager
from TTSManager import TTSManager
from config import MODEL_LIST

model_manager = ModelManager()
tts_manager = TTSManager(model_manager)

# Register TTSManager as an observer of ModelManager
model_manager.attach(tts_manager)

model_manager.model_init(MODEL_LIST)

# Now, when you load or unload a model in ModelManager, TTSManager will be notified
# model_manager.load_model("path_to_model")
# model_manager.unload_model(0)