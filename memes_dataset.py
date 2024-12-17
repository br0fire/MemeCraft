from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader

ImageFile.LOAD_TRUNCATED_IMAGES = True
import os


def meme_collate_fn(batch):
    new_batch = [item for item in batch]

    return new_batch


class MemesDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        for dir in os.listdir(self.root_dir):
            dir_path = os.path.join(self.root_dir, dir)
            self.images += [os.path.join(dir_path, image) for image in os.listdir(dir_path)
                            if os.path.isfile(os.path.join(dir_path, image)) and (image.endswith('.jpg') \
                                                                                  or image.endswith(
                            '.png') or image.endswith('.jpeg') or image.endswith('.JPG'))]

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
