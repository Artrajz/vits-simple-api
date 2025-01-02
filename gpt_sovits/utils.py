import logging
import subprocess

import numpy as np


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
    ffmpeg_process = subprocess.Popen(
        ffmpeg_command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    out, info = ffmpeg_process.communicate(input=raw_audio)

    # 将管道输出解码为 NumPy 数组
    processed_audio = np.frombuffer(out, np.float32)

    return processed_audio
