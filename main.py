import argparse

from PIL import Image
import torchvision
import argparse
import logging
from transformers import CLIPProcessor, CLIPModel


logger = logging.getLogger(__name__)

def pipeline(prompt: str):
    logger.info("Start meme generation")




    # model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    # processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    #
    # url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # image = Image.open(requests.get(url, stream=True).raw)
    #
    # inputs = processor(text=["a photo of a cat", "a photo of a dog"]





if __name__ == '__main__':
    logger.info("Start meme generation")
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--propmt', default='', type=str)
    args = parser.parse_args()

    pipeline(args.prompt)