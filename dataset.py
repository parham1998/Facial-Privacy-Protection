# =============================================================================
# Import required libraries
# =============================================================================
import os
from PIL import Image
from torch.utils.data import Dataset


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                fname = fname.split('.')[0]
                images.append((fname, path))
    return images


class ImageDataset(Dataset):
    def __init__(self,
                 source_root,
                 source_transform=None):
        self.source_paths = sorted(make_dataset(source_root))
        self.source_transform = source_transform

    def __len__(self):
        return len(self.source_paths)

    def __getitem__(self, index):
        fname, from_path = self.source_paths[index]
        image = Image.open(from_path).convert('RGB')
        #
        if self.source_transform:
            image = self.source_transform(image)

        return fname, image
