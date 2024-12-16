from torch.utils.data import Dataset, DataLoader
import PIL
import os

class MemesDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        for dir in os.listdir(self.root_dir):
            dir_path = os.path.join(self.root_dir, dir)
            self.images += [os.path.join(dir_path, image) for image in os.listdir(dir_path) 
                       if os.path.isfile(os.path.join(dir_path, image)) and (image.endswith('.jpg') \
                       or image.endswith('.png') or image.endswith('.jpeg')or image.endswith('.JPG'))]
        
        if len(self.images) == 0:
            raise ValueError('Dataset could not be empty')
          
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = PIL.Image.open(self.images[idx])
        if self.transform:
            image = self.transform(image)

        return image
    
def meme_collate_fn(batch):
    new_batch = [item for item in batch]

    return new_batch