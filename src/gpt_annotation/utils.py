import base64
import logging
import os
import time

def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def setup_logger(base_path: str) -> logging.Logger:
    """Setup a unified logger that writes to console and file."""
    log_dir = os.path.join(base_path, "logs")
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f"run_{time.strftime('%Y%m%d_%H%M%S')}.log")

    logger = logging.getLogger("bitefinder")
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers if setup_logger is called multiple times
    if not logger.handlers:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        ch = logging.StreamHandler()

        formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S")
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger
 