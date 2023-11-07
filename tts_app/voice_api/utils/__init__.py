import os


def clean_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        # 如果是文件，则删除文件。如果是文件夹则跳过。
        if os.path.isfile(file_path):
            os.remove(file_path)


def check_is_none(item) -> bool:
    # none -> True, not none -> False
    return item is None or (isinstance(item, str) and str(item).isspace()) or str(item) == ""


def save_audio(audio, path):
    with open(path, "wb") as f:
        f.write(audio)