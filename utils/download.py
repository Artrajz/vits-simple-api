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


def download_file(url, dest_path):
    logging.info(f"Downloading: {url}")
    with TqdmUpTo(unit="B", unit_scale=True, unit_divisor=1024, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, dest_path, reporthook=t.update_to)


def verify_md5(file_path, expected_md5):
    md5 = hashlib.md5(file_path.read_bytes()).hexdigest()
    if md5 != expected_md5:
        return False, f"MD5 mismatch: {md5} != {expected_md5}"
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


def download_and_verify(urls, target_path, expected_md5=None, extract_destination=None):
    for url in urls:
        try:
            download_file(url, target_path)
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

    success, message = download_and_verify(URLS, TARGET_PATH, EXPECTED_MD5, EXTRACT_DESTINATION)
    print(message)
