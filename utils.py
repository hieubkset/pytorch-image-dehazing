import os
import os.path
import torch
import sys
import numpy as np
from tensorboardX import SummaryWriter
from torchvision import transforms


def tensor_to_rgb(x):
    output = x.cpu()
    output = output.data.squeeze(0).numpy()
    output = (output + 1.0) * 127.5
    output = output.clip(0, 255).transpose(1, 2, 0)
    return output


def rgb_to_tensor(x):
    output = (transforms.ToTensor()(x) - 0.5) / 0.5
    return output


def get_file_paths(imgdir):
    file_paths = []
    for file_name in os.listdir(imgdir):
        file_paths.append(os.path.join(imgdir, file_name))
    file_paths = sorted(file_paths)
    return file_paths


class SaveData:
    def __init__(self, save_dir, exp, finetuning):
        self.exp_dir = os.path.join(save_dir, exp)

        if not finetuning:
            if os.path.exists(self.exp_dir):
                os.system('rm -rf ' + self.exp_dir)
                print("! Remove a folder: " + self.exp_dir)

        if not os.path.exists(self.exp_dir):
            os.makedirs(self.exp_dir)

        self.model_dir = os.path.join(self.exp_dir, 'model')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        self.logfile = open(self.exp_dir + '/logs.txt', 'a')

        tensorboard_dir = os.path.join(self.exp_dir, 'tb')
        if not os.path.exists(tensorboard_dir):
            os.makedirs(tensorboard_dir)
        self.writer = SummaryWriter(tensorboard_dir)

    def save_params(self, args):
        with open(self.exp_dir + '/params.txt', 'w') as params_file:
            params_file.write(str(args.__dict__) + "\n")

    def save_model(self, model, epoch):
        torch.save(model.state_dict(), os.path.join(self.model_dir, 'model_lastest.pt'))
        # torch.save(model.state_dict(), os.path.join(self.model_dir, 'model_%04d.pt' % epoch))
        torch.save(model, os.path.join(self.model_dir, 'model_obj.pt'))
        torch.save(epoch, os.path.join(self.model_dir, 'last_epoch.pt'))

    def save_log(self, log):
        sys.stdout.flush()
        self.logfile.write(log + '\n')
        self.logfile.flush()

    def load_model(self, model):
        model.load_state_dict(torch.load(self.model_dir + '/model_lastest.pt'))
        last_epoch = torch.load(self.model_dir + '/last_epoch.pt')
        print("Load mode_status from {}/model_lastest.pt, epoch: {}".format(self.model_dir, last_epoch))
        return model, last_epoch

    def add_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)


class ImagePool:
    def __init__(self, pool_size=50):
        self.pool_size = pool_size
        if pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, image):
        if self.pool_size == 0:
            return image
        if self.num_imgs < self.pool_size:
            self.images.append(image.clone())
            self.num_imgs += 1
            return image
        else:
            if np.random.uniform(0, 1) > 0.5:
                random_id = np.random.randint(self.pool_size, size=1)[0]
                tmp = self.images[random_id].clone()
                self.images[random_id] = image.clone()
                return tmp
            else:
                return image
