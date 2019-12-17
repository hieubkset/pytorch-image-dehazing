import os
import random
import torch.utils.data as data
from PIL import Image
from torchvision import transforms
from utils import rgb_to_tensor


def augment(img_input, img_target):
    degree = random.choice([0, 90, 180, 270])
    if degree != 0:
        img_input = transforms.functional.rotate(img_input, degree)
        img_target = transforms.functional.rotate(img_target, degree)

    return img_input, img_target


def get_patch(img_input, img_target):
    w, h = img_input.size
    if w >= 2048 and h >= 2048:
        choice = random.choice([1, 2, 3])
        if choice == 1:
            p_x = p_y = 1024
        elif choice == 2:
            p_x = 1024
            p_y = 2048
        else:
            p_x = p_y = 2048
    else:
        p_x = p_y = 1024
    x = random.randrange(0, w - p_x + 1)
    y = random.randrange(0, h - p_y + 1)
    img_input = img_input.crop((x, y, x + p_x, y + p_y))
    img_target = img_target.crop((x, y, x + p_x, y + p_y))
    if p_x > 1024 or p_y > 1024:
        img_input = img_input.resize((1024, 1024))
        img_target = img_target.resize((1024, 1024))
    return img_input, img_target


def is_large_image(image_path):
    w, h = Image.open(image_path).size
    return w >= 1024 and h >= 1024


def get_file_paths(folder):
    file_paths = []
    for file_name in os.listdir(folder):
        file_path = os.path.join(folder, file_name)
        if is_large_image(file_path):
            file_paths.append(file_path)
    file_paths = sorted(file_paths)
    return file_paths


class MyDataset(data.Dataset):
    def __init__(self, data_dir, is_train=False):
        super(MyDataset, self).__init__()
        self.is_train = is_train

        hazy_dir = os.path.join(data_dir, 'Hazy')
        target_dir = os.path.join(data_dir, 'GT')

        self.input_file_paths = get_file_paths(hazy_dir)
        self.target_file_paths = get_file_paths(target_dir)

        self.n_samples = len(self.input_file_paths)

    def get_img_pair(self, idx):
        img_input = Image.open(self.input_file_paths[idx]).convert('RGB')
        img_target = Image.open(self.target_file_paths[idx]).convert('RGB')

        return img_input, img_target

    def __getitem__(self, idx):
        img_input, img_target = self.get_img_pair(idx)

        if self.is_train:
            img_input, img_target = get_patch(img_input, img_target)
            img_input, img_target = augment(img_input, img_target)

        img_input = rgb_to_tensor(img_input)
        img_target = rgb_to_tensor(img_target)

        return img_input, img_target

    def __len__(self):
        return self.n_samples
