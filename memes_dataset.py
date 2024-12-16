from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os


def meme_collate_fn(batch):
    new_batch = [item for item in batch]

    return new_batch


class MemesDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = [os.path.join(self.root_dir, image) for image in os.listdir(root_dir) 
                       if os.path.isfile(os.path.join(self.root_dir, image)) and (image.endswith('.jpg') or image.endswith('.png'))]
        if len(self.images) == 0:
            raise ValueError('Dataset could not be empty')
          
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx])
        if self.transform:
            image = self.transform(image)

        return image

    def create_dataloader(self, batch_size: int, shuffle: bool = False, num_workers: int = 0) -> DataLoader:
        return DataLoader(self,
                          batch_size=batch_size,
                          shuffle=shuffle,
                          collate_fn=meme_collate_fn,
                          num_workers=num_workers
                          )
