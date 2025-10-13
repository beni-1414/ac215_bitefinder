import os
import json
import time
import traceback
from openai import OpenAI
from src.gpt_annotation.utils import encode_image
from src.gpt_annotation.prompt import PROMPT_TEMPLATE

class BiteAnnotator:
    def __init__(self, client: OpenAI, image_dir: str, output_file: str, model_name: str, bite_type: str, logger):
        self.client = client
        self.image_dir = image_dir
        self.output_file = output_file
        self.model_name = model_name
        self.bite_type = bite_type
        self.logger = logger

        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)

    def already_processed(self, image_name: str) -> bool:
        """Check if the image has already been processed. This prevents re-processing images in case of interruptions.
        Args:
            image_name (str): Name of the image file.
        """
        if not os.path.exists(self.output_file):
            return False
        with open(self.output_file, "r", encoding="utf-8") as f:
            return any(image_name in line for line in f)

    def process_all(self, limit: int | None = None):
        """
        Process all images in the directory, or up to the specified limit.
        Args:
            limit (int | None): Maximum number of images to process. If None, process all.
        """
        images = sorted(os.listdir(self.image_dir))
        if limit:
            images = images[:limit]
        total = len(images)
        self.logger.info(f"Starting batch of {total} images using {self.model_name}")

        for idx, image in enumerate(images, 1):
            try:
                if self.already_processed(image):
                    self.logger.info(f"Skipping {image} (already processed)")
                    continue

                image_path = os.path.join(self.image_dir, image)
                image_data = encode_image(image_path)
                prompt = PROMPT_TEMPLATE.format(bite_type=self.bite_type)

                response = self.client.chat.completions.create(
                    model=self.model_name,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": "You are an expert medical image annotator."},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
                            ],
                        },
                    ],
                    temperature=0,
                )

                content = response.choices[0].message.content

                try:
                    result = json.loads(content)
                except Exception as e:
                    self.logger.error(f"[{image}] JSON parse error: {e}")
                    continue

                with open(self.output_file, "a", encoding="utf-8") as of:
                    of.write(json.dumps({"image": image, "result": result}, ensure_ascii=False) + "\n")

                self.logger.info(f"[{idx}/{total}] ✅ {image} processed successfully")

            except Exception as e:
                self.logger.error(f"[{idx}/{total}] ⚠️ Error processing {image}: {e}")
                self.logger.debug(traceback.format_exc())

            time.sleep(1.2)

        self.logger.info("✅ Batch processing complete.")
