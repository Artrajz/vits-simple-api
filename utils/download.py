import logging
import os
import hashlib
import tarfile
import zipfile
from pathlib import Path

import requests
from py7zr import SevenZipFile
from tqdm import tqdm
from config import ABS_PATH


def _download_file(url, dest_path, max_retry=1):
    logging.info(f"Downloading: {url}")

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }

    try:
        response = requests.head(url, headers=headers, allow_redirects=True, timeout=10)
        if response.status_code >= 400:
            logging.error(f"Failed to connect to {url}, status code: {response.status_code}")
            return False, f"Failed to connect, status code: {response.status_code}"
    except Exception as e:
        logging.error(f"Failed to get file size for {url}: {e}")
        return False, f"Request timeout: {e}"

    total_size = int(response.headers.get('content-length', 0))
    file_size = os.path.getsize(dest_path) if os.path.exists(dest_path) else 0

    if file_size == total_size:
        logging.info(f"File {dest_path} already downloaded and complete.")
        return True, "File already downloaded and complete."
    elif file_size > total_size:
        logging.warning(f"Local file size {file_size} exceeds server file size {total_size}. Removing local file.")
        os.remove(dest_path)
        if max_retry <= 0:
            return False, "Local file size exceeds server file size."
        return _download_file(url, dest_path, max_retry=max_retry - 1)

    headers['Range'] = f'bytes={file_size}-' if file_size > 0 else None

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    relative_path = os.path.relpath(dest_path, ABS_PATH)
    chunk_size = 1024 * 1024  # 1MB

    try:
        with requests.get(url, headers=headers, stream=True, timeout=10) as response, open(dest_path, 'ab') as file, tqdm(
                total=total_size,
                initial=file_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
                desc=f"Downloading: {relative_path or url.split('/')[-1]}",
        ) as progress:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    file.write(chunk)
                    progress.update(len(chunk))

        logging.info(f"Download completed: {dest_path}")
        return True, "Download completed."
    except Exception as e:
        logging.error(f"Error during downloading {url}: {e}")
        if max_retry > 0:
            logging.info(f"Retrying download ({max_retry} retries left)...")
            return _download_file(url, dest_path, max_retry=max_retry - 1)
        return False, f"Download failed: {e}"


def verify_md5(file_path, expected_md5):
    md5 = hashlib.md5(file_path.read_bytes()).hexdigest()
    if md5 != expected_md5:
        return False, f"MD5 mismatch: {md5} != {expected_md5}"
    return True, ""


def verify_sha256(file_path, expected_sha256):
    sha256 = hashlib.sha256(file_path.read_bytes()).hexdigest()
    if sha256 != expected_sha256:
        return False, f"SHA256 mismatch: {sha256} != {expected_sha256}"
    return True, ""


def extract_file(file_path, destination=None):
    """
    Extract a compressed file based on its extension.
    If destination is not specified, it will be extracted to its parent directory.
    """
    if destination is None:
        destination = Path(file_path).parent

    logging.info(f"Extracting to {destination}")

    if file_path.endswith('.zip'):
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(destination)
    elif file_path.endswith('.tar.gz'):
        with tarfile.open(file_path, 'r:gz') as tar_ref:
            tar_ref.extractall(destination)
    elif file_path.endswith('.tar.bz2'):
        with tarfile.open(file_path, 'r:bz2') as tar_ref:
            tar_ref.extractall(destination)
    elif file_path.endswith('.7z'):
        with SevenZipFile(file_path, mode='r') as z:
            z.extractall(destination)
    else:
        logging.error(f"Unsupported compression format for file {file_path}")


def download_file(urls, target_path, extract_destination=None, expected_md5=None, expected_sha256=None):
    if os.path.exists(target_path):
        success_msg = "File already exists and verified successfully!"
        if expected_md5 is not None:
            success, message = verify_md5(Path(target_path), expected_md5)
            if success:
                return True, success_msg

        if expected_sha256 is not None:
            success, message = verify_sha256(Path(target_path), expected_sha256)
            if success:
                return True, success_msg

        # If it's a compressed file and the target_path already exists, skip the download
        if extract_destination and target_path.endswith(('.zip', '.tar.gz', '.tar.bz2', '.7z')):
            extract_file(target_path, extract_destination)
            os.remove(target_path)
            return True, success_msg

    is_download = False
    for url in urls:
        try:
            is_download, _ = _download_file(url, target_path)
            if is_download:
                break
        except Exception as error:
            logging.error(f"downloading from URL {url}: {error}")

    if not is_download:
        return False, "Error downloading from all provided URLs."

    if expected_md5 is not None:
        success, message = verify_md5(Path(target_path), expected_md5)
        if not success:
            return False, message

    if expected_sha256 is not None:
        success, message = verify_sha256(Path(target_path), expected_sha256)
        if not success:
            return False, message

    # If it's a compressed file, extract it
    if target_path.endswith(('.zip', '.tar.gz', '.tar.bz2', '.7z')):
        extract_file(target_path, extract_destination)
        os.remove(target_path)

    return True, "File downloaded, verified, and extracted successfully!"


if __name__ == "__main__":
    import logger

    URL = [
        "https://hf-mirror.com/hfl/chinese-roberta-wwm-ext-large/resolve/main/pytorch_model.bin",
    ]
    TARGET_PATH = r"E:\work\vits-simple-api\data\bert\chinese-roberta-wwm-ext-large/pytorch_model1.bin"
    EXPECTED_MD5 = None
    EXTRACT_DESTINATION = None

    print(download_file(URL, TARGET_PATH, EXPECTED_MD5, EXTRACT_DESTINATION))
