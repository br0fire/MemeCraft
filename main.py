from memes_dataset import MemesDataset
from transformers import CLIPProcessor, CLIPModel
import torch
import numpy as np
import argparse
import logging
from pathlib import Path
import os

logger = logging.getLogger(__name__)


def pipeline(prompt: str):
    logger.info("Start meme generation")

    data_images = MemesDataset(root_dir="./img")
    dataloader = data_images.create_dataloader(batch_size=4, shuffle=False)

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    scores = []
    for batch in dataloader:
        inputs = processor(text=[prompt], images=batch, return_tensors="pt", padding=True)

        with torch.no_grad():
            outputs = model(**inputs)

        img_emb = outputs.image_embeds
        txt_emb = outputs.text_embeds
        clip_score = txt_emb @ img_emb.T
        scores.extend(clip_score.squeeze().tolist())

    top_scores_indexes = np.argsort(scores)[::-1][:3]
    top_images = [data_images[ind] for ind in top_scores_indexes]

    os.makedirs("./mem_img", exist_ok=True)
    for i, img in enumerate(top_images):
        img.save(Path("./mem_img") / f"mem_image_{i}")


if __name__ == '__main__':
    logger.info("Start meme generation")
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--prompt', default='', type=str)
    args = parser.parse_args()

    pipeline(args.prompt)