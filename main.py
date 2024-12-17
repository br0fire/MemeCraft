import argparse
import os
import warnings
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from tqdm.auto import tqdm
from transformers import CLIPProcessor, CLIPModel

from meme_caption_generator import MemeCaptionGenerator
from memes_dataset import MemesDataset
from text_on_image import add_caption_to_image


def pipeline(prompt: str | None, topic: str | None, data: str):
    logger.info("Start meme generation")

    data_images = MemesDataset(root_dir=data)
    dataloader = data_images.create_dataloader(batch_size=4, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    model = CLIPModel.from_pretrained("wkcn/TinyCLIP-ViT-61M-32-Text-29M-LAION400M").to(device)
    # processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("wkcn/TinyCLIP-ViT-61M-32-Text-29M-LAION400M")
    if len(prompt) == 0:
        generator = MemeCaptionGenerator()
        response = generator.generate_caption(topic=topic)
    else:
        response = prompt
    logger.info(f"Caption: {response}")

    scores = []
    logger.info("Data Processing...")
    for batch in tqdm(dataloader):
        inputs = processor(text=[response], images=batch, return_tensors="pt", padding=True)
        inputs['input_ids'] = inputs['input_ids'].to(device)
        inputs['attention_mask'] = inputs['attention_mask'].to(device)
        inputs['pixel_values'] = inputs['pixel_values'].to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        img_emb = outputs.image_embeds
        txt_emb = outputs.text_embeds
        clip_score = txt_emb @ img_emb.T
        scores.extend(clip_score.squeeze(0).cpu().tolist())

    top_scores_indexes = np.argsort(scores)[::-1][:3]
    top_images = [data_images[ind] for ind in top_scores_indexes]

    os.makedirs("./mem_img", exist_ok=True)
    for i, img in enumerate(top_images):
        add_caption_to_image(img, response, Path("./mem_img") / f"mem_image_{i}.jpg")


if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--prompt', default='', type=str)
    parser.add_argument('-t', '--topic', default='', type=str)
    parser.add_argument('-d', '--data', default='./datasets', type=str)
    args = parser.parse_args()

    pipeline(args.prompt, args.topic, args.data)
