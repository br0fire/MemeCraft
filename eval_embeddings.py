import torch
from tqdm.auto import tqdm
from transformers import AutoProcessor, CLIPVisionModelWithProjection

from memes_dataset import MemesDataset


data_images = MemesDataset(root_dir="./datasets")
dataloader = data_images.create_dataloader(batch_size=4, shuffle=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
img_emb = []

for batch in tqdm(dataloader):
    inputs = processor(images=batch, return_tensors="pt")
    inputs['pixel_values'] = inputs['pixel_values'].to(device)
    with torch.no_grad():
        outputs = model(**inputs)

img_emb.append(outputs.image_embeds.cpu())

result = torch.cat(img_emb)
torch.save(result, "result.pt")
