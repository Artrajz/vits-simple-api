import logging
import subprocess

import librosa
import numpy as np

import config


class DictToAttrRecursive(dict):
    def __init__(self, input_dict):
        super().__init__(input_dict)
        for key, value in input_dict.items():
            if isinstance(value, dict):
                value = DictToAttrRecursive(value)
            self[key] = value
            setattr(self, key, value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DictToAttrRecursive(value)
        super(DictToAttrRecursive, self).__setitem__(key, value)
        super().__setattr__(key, value)

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")


# def load_audio(file, sr=16000):
#     try:
#         y, sr = librosa.load(file, sr=sr, dtype=np.float32)
#     except Exception as e:
#         raise RuntimeError(f"Failed to load audio: {e}")
#
#     return y.flatten(), sr

# import ffmpeg
# import numpy as np
#
#
# def load_audio(file, sr):
#     try:
#         # https://github.com/openai/whisper/blob/main/whisper/audio.py#L26
#         # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
#         # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
#         file = (
#             file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
#         )  # 防止小白拷路径头尾带了空格和"和回车
#         out, _ = (
#             ffmpeg.input(file, threads=0)
#             .output("-", format="f32le", acodec="pcm_f32le", ac=1, ar=sr)
#             .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
#         )
#     except Exception as e:
#         raise RuntimeError(f"Failed to load audio: {e}")
#
#     return np.frombuffer(out, np.float32).flatten()

def speed_change(input_audio: np.ndarray, speed_factor: float, sr: int):
    """这里修改为使用原数据float32变速，GPT-SoVITS官方为int16"""

    raw_audio = input_audio.tobytes()

    ffmpeg_command = [
        # f'{BASE_DIR}/bin/ffmpeg',
        'ffmpeg',
        '-f', 'f32le',
        '-ar', str(sr),
        '-ac', '1',
        '-i', 'pipe:',
        '-filter:a', f'atempo={speed_factor}',
        '-f', 'f32le',
        '-acodec', 'pcm_f32le',
        'pipe:'
    ]
    try:
        ffmpeg_process = subprocess.Popen(
            ffmpeg_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        out, info = ffmpeg_process.communicate(input=raw_audio)
    except Exception as e:
        logging.error(e)
        logging.error(info)

    # 将管道输出解码为 NumPy 数组
    processed_audio = np.frombuffer(out, np.float32)

    return processed_audio
