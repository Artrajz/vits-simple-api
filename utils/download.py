import logging
import os
import hashlib
import tarfile
import urllib.request
import zipfile

from tqdm import tqdm
from pathlib import Path
from logger import logger
from py7zr import SevenZipFile


class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def _download_file(url, dest_path):
    logging.info(f"Downloading: {url}")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }

    if os.path.exists(dest_path):
        file_size = os.path.getsize(dest_path)
        headers['Range'] = f'bytes={file_size}-'

    request = urllib.request.Request(url, headers=headers)

    response = urllib.request.urlopen(request)
    if response.geturl() != url:
        return _download_file(response.geturl(), dest_path)

    total_size = int(response.headers['Content-Length'])

    with open(dest_path, 'ab') as file, tqdm(total=total_size, unit='B', unit_scale=True, unit_divisor=1024, miniters=1,
                                             desc=url.split('/')[-1]) as t:
        chunk_size = 1024 * 1024  # 1MB
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            file.write(chunk)
            t.update(len(chunk))


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
        if expected_md5 is not None:
            success, message = verify_md5(Path(target_path), expected_md5)
            if not success:
                os.remove(target_path)
                return False, message

        if expected_sha256 is not None:
            success, message = verify_sha256(Path(target_path), expected_sha256)
            if not success:
                os.remove(target_path)
                return False, message

        # If it's a compressed file and the target_path already exists, skip the download
        if extract_destination and target_path.endswith(('.zip', '.tar.gz', '.tar.bz2', '.7z')):
            extract_file(target_path, extract_destination)
            os.remove(target_path)

        return True, "File already exists and verified successfully!"
    
    for url in urls:
        try:
            _download_file(url, target_path)
            break
        except Exception as error:
            logger.error(f"downloading from URL {url}: {error}")

    else:  # This else is tied to the for loop, and executes if no download is successful
        return False, "Error downloading from all provided URLs."

    if expected_md5 is not None:
        success, message = verify_md5(Path(target_path), expected_md5)
        if not success:
            os.remove(target_path)
            return False, message
        
    if expected_sha256 is not None:
        success, message = verify_sha256(Path(target_path), expected_sha256)
        if not success:
            os.remove(target_path)
            return False, message

    # If it's a compressed file, extract it
    if target_path.endswith(('.zip', '.tar.gz', '.tar.bz2', '.7z')):
        extract_file(target_path, extract_destination)
        os.remove(target_path)

    return True, "File downloaded, verified, and extracted successfully!"


if __name__ == "__main__":
    URLS = [
        "YOUR_PRIMARY_URL_HERE",
        "YOUR_FIRST_BACKUP_URL_HERE",
        # ... you can add more backup URLs as needed
    ]
    TARGET_PATH = ""
    EXPECTED_MD5 = ""
    EXTRACT_DESTINATION = ""

    success, message = download_file(URLS, TARGET_PATH, EXPECTED_MD5, EXTRACT_DESTINATION)
    print(message)
