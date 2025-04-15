import nltk
import logging
import ssl
import os
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENV_NLTK_DATA_DIR = os.path.join(project_root, 'venv', 'nltk_data')

# --- NLTK Data Packages to Download ---
PACKAGES = ['punkt', 'punkt_tab']
# ------------------------------------

def ensure_dir_exists(path):
    """Creates a directory if it doesn't exist."""
    if not os.path.exists(path):
        logger.info(f"Creating directory: {path}")
        try:
            os.makedirs(path)
        except OSError as e:
            logger.error(f"Failed to create directory {path}: {e}")
            return False
    return True

def download_packages(package_list, download_dir):
    """Downloads NLTK packages to a specific directory."""
    logger.info(f"Target NLTK download directory: {download_dir}")
    if not ensure_dir_exists(download_dir):
        return 

    if download_dir not in nltk.data.path:
         nltk.data.path.insert(0, download_dir) 
         logger.info(f"Added '{download_dir}' to start of nltk.data.path for this script.")

    logger.info(f"Attempting to download NLTK packages: {package_list} to {download_dir}")
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
        logger.warning("Attempting to bypass SSL verification for NLTK download.")

    downloaded_count = 0
    failed_packages = []
    for package in package_list:
        try:
            logger.info(f"Checking/Downloading '{package}' to {download_dir}...")
            if nltk.download(package, download_dir=download_dir, quiet=False, force=True, raise_on_error=True):
                logger.info(f"Successfully downloaded/updated '{package}' in {download_dir}.")
                downloaded_count += 1
            else:
                 logger.info(f"'{package}' verification passed (likely already up-to-date) in {download_dir}.")
        except ValueError as ve:
             logger.error(f"Failed to download '{package}'. It might not be a valid NLTK package ID. Error: {ve}")
             failed_packages.append(package)
        except Exception as e:
            logger.error(f"Failed to download or verify '{package}'. Error: {e}")
            failed_packages.append(package)

    logger.info(f"NLTK package check/download process finished for directory {download_dir}.")
    if failed_packages:
        logger.warning(f"Could not download the following packages: {failed_packages}")

if __name__ == "__main__":
    download_packages(PACKAGES, VENV_NLTK_DATA_DIR)