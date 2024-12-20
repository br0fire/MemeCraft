import argparse
import os
import warnings
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from transformers import AutoTokenizer, CLIPTextModelWithProjection

from meme_caption_generator import MemeCaptionGenerator
from memes_dataset import MemesDataset
from text_on_image import add_caption_to_image


def pipeline(prompt: str | None, topic: str | None, data: str):
    logger.info("Start meme generation")

    data_images = MemesDataset(root_dir=data)
    clip_model = "openai/clip-vit-large-patch14"
    model = CLIPTextModelWithProjection.from_pretrained(clip_model)
    tokenizer = AutoTokenizer.from_pretrained(clip_model)

    if len(prompt) == 0:
        generator = MemeCaptionGenerator()
        response = generator.generate_caption(topic=topic)
    else:
        response = prompt

    logger.info(f"Caption: {response}")
    logger.info("Data Processing...")

    inputs = tokenizer([response], return_tensors="pt", padding=True)
    outputs = model(**inputs)

    txt_emb = outputs.text_embeds
    img_emb = torch.load("result.pt")
    clip_score = txt_emb @ img_emb.T

    top_scores_indexes = np.argsort(clip_score.flatten().detach().numpy())[::-1][:5]
    top_images = [data_images[ind] for ind in top_scores_indexes]

    os.makedirs("./mem_img", exist_ok=True)

    for i, img in enumerate(top_images):
        add_caption_to_image(
            img, response, Path("./mem_img") / f"top_{i+1}_{response}.jpg"
        )

    logger.info("Memes saved!")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--prompt", default="", type=str)
    parser.add_argument("-t", "--topic", default="", type=str)
    parser.add_argument("-d", "--data", default="./datasets", type=str)
    args = parser.parse_args()

    pipeline(args.prompt, args.topic, args.data)
