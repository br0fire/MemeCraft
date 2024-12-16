from memes_dataset import MemesDataset
from transformers import CLIPProcessor, CLIPModel
from meme_caption_generator import  MemeCaptionGenerator
from text_on_image import add_caption_to_image
import torch
import numpy as np
import argparse
import logging
from pathlib import Path
import os

logger = logging.getLogger(__name__)


def pipeline(prompt: str, topic: str | None):
    logger.info("Start meme generation")

    data_images = MemesDataset(root_dir="./img")
    dataloader = data_images.create_dataloader(batch_size=4, shuffle=False)

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    generator = MemeCaptionGenerator()

    response = generator.generate_caption(topic=topic) if len(prompt) == 0 else prompt

    scores = []
    for batch in dataloader:
        inputs = processor(text=[response], images=batch, return_tensors="pt", padding=True)

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
        add_caption_to_image(img, prompt, Path("./mem_img") / f"mem_image_{i}.jpg")

    generator = MemeCaptionGenerator()
    while True:
        s = input("Enter a topic of a meme: ").strip()
        if len(s) == 0:
            print(generator.generate_caption(), end="\n\n")
        else:
            print(generator.generate_caption(topic=s), end="\n\n")


if __name__ == '__main__':
    logger.info("Start meme generation")
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--prompt', default='', type=str)
    parser.add_argument('-t', '--topic', default='', type=str)
    args = parser.parse_args()

    pipeline(args.prompt, args.topic)
