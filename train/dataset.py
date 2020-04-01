import numpy as np
import os

from PIL import Image

from torch.utils.data import Dataset

EXTENSIONS = ['.jpg', '.png']

def load_image(file):
    return Image.open(file)

def is_image(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS)

def is_label(filename):
    return filename.endswith("_labelIds.png")

def image_path(root, basename, extension):
    return os.path.join(root, f'{basename}{extension}')

def image_path_city(root, name):
    return os.path.join(root, f'{name}')

def image_basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])

class VOC12(Dataset):

    def __init__(self, root, input_transform=None, target_transform=None):
        self.images_root = os.path.join(root, 'images')
        self.labels_root = os.path.join(root, 'labels')

        self.filenames = [image_basename(f)
            for f in os.listdir(self.labels_root) if is_image(f)]
        self.filenames.sort()

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        filename = self.filenames[index]

        with open(image_path(self.images_root, filename, '.jpg'), 'rb') as f:
            image = load_image(f).convert('RGB')
        with open(image_path(self.labels_root, filename, '.png'), 'rb') as f:
            label = load_image(f).convert('P')

        if self.input_transform is not None:
            image = self.input_transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label

    def __len__(self):
        return len(self.filenames)




class cityscapes(Dataset):

    def __init__(self, root, co_transform=None, subset='train',rate=100):
        self.images_root = os.path.join(root, 'leftImg8bit/')
#         self.labels_root = os.path.join(root, '../gtFine_trainvaltest/gtFine/')
        
        self.images_root += subset
        self.labels_root = "/esat/toyota/trace/deeplearning/datasets_public/cityscapes/gtFine_trainvaltest/gtFine/"+subset

#         print("This is the subset!")
#         print(subset)/
        print (self.images_root)
        print (self.labels_root)

        #self.filenames = [image_basename(f) for f in os.listdir(self.images_root) if is_image(f)]
        file_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.images_root)) for f in fn if is_image(f)]
        file_names.sort()

        if (rate > 100 or rate <=0):
            raise ValueError("Rate must be between 0 and 100")
        self.actual_rate = int(len(file_names)*(rate/100))
        self.filenames = file_names[0:self.actual_rate]
        print(len(self.filenames))
#         print(self.filenames)
        #[os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(".")) for f in fn]
        #self.filenamesGt = [image_basename(f) for f in os.listdir(self.labels_root) if is_image(f)]
        file_namesGt = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.labels_root)) for f in fn if is_label(f)]
        file_namesGt.sort()
        self.filenamesGt = file_namesGt[0:self.actual_rate]
        print(len(self.filenamesGt))
#         print(self.filenamesGt)
        self.co_transform = co_transform # ADDED THIS


    def __getitem__(self, index):
        filename = self.filenames[index]
        filenameGt = self.filenamesGt[index]
        
#         print("this is me from dataset")
#         print(filenameGt)

        with open(image_path_city(self.images_root, filename), 'rb') as f:
            image = load_image(f).convert('RGB')
        with open(image_path_city(self.labels_root, filenameGt), 'rb') as f:
            label = load_image(f).convert('P')

        if self.co_transform is not None:
            image, label = self.co_transform(image, label)

        return image, label

    def __len__(self):
#         print("this is me from dataset")
#         print(self.filenameGt)
        return len(self.filenames)

