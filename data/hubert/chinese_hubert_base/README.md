---
license: mit
---
Pretrained on 10k hours WenetSpeech L subset. More details in  [TencentGameMate/chinese_speech_pretrain](https://github.com/TencentGameMate/chinese_speech_pretrain)

This model does not have a tokenizer as it was pretrained on audio alone. 
In order to use this model speech recognition, a tokenizer should be created and the model should be fine-tuned on labeled text data.

python package:
transformers==4.16.2

```python


import torch
import torch.nn.functional as F
import soundfile as sf

from transformers import (
    Wav2Vec2FeatureExtractor,
    HubertModel,
)


model_path=""
wav_path=""

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
model = HubertModel.from_pretrained(model_path)

# for pretrain: Wav2Vec2ForPreTraining
# model = Wav2Vec2ForPreTraining.from_pretrained(model_path)

model = model.to(device)
model = model.half()
model.eval()

wav, sr = sf.read(wav_path)
input_values = feature_extractor(wav, return_tensors="pt").input_values
input_values = input_values.half()
input_values = input_values.to(device)

with torch.no_grad():
    outputs = model(input_values)
    last_hidden_state = outputs.last_hidden_state


```