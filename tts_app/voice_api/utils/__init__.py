def check_is_none(item) -> bool:
    # none -> True, not none -> False
    return item is None or (isinstance(item, str) and str(item).isspace()) or str(item) == ""


def save_audio(audio, path):
    with open(path, "wb") as f:
        f.write(audio)
