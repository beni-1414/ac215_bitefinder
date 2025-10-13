import os
from dotenv import load_dotenv
from openai import OpenAI
from src.gpt_annotation.utils import setup_logger
from src.gpt_annotation.gpt_annotator import BiteAnnotator

BASE_PATH = 'C:/Users/bfite/Desktop/repos/ac215_bitefinder'
IMAGE_DIR = os.path.join(BASE_PATH, "data/testing/ants")
OUTPUT_FILE = os.path.join(BASE_PATH, "results/annotation_results.jsonl")
MODEL_NAME = "gpt-4.1"
BITE_TYPE = ["ant"]

load_dotenv(os.path.join(BASE_PATH, ".env"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def main():
    logger = setup_logger(BASE_PATH)
    client = OpenAI(api_key=OPENAI_API_KEY)

    for bite in BITE_TYPE:
        logger.info(f"Starting annotation for bite type: {bite}")

        annotator = BiteAnnotator(
            client=client,
            image_dir=IMAGE_DIR,
            output_file=OUTPUT_FILE,
            model_name=MODEL_NAME,
            bite_type=bite,
            logger=logger,
        )

        annotator.process_all(limit=2)

if __name__ == "__main__":
    main()