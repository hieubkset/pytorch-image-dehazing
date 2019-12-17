import os
import argparse
from model import Generator, Discriminator
from utils import SaveData, ImagePool
from data import MyDataset
from vgg16 import Vgg16

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd import Variable

from tqdm import tqdm

parser = argparse.ArgumentParser(description='image-dehazing')
parser.add_argument('--data_dir', type=str, default='dataset/indoor',
                    help='dataset directory')
parser.add_argument('--save_dir', default='results', help='data save directory')
parser.add_argument('--batch_size', type=int, default=1, help='batch size for training')
parser.add_argument('--n_threads', type=int, default=8, help='number of threads for data loading')

# model
parser.add_argument('--exp', default='Net1', help='model to select')

# optimization
parser.add_argument('--p_factor', type=float, default=0.5, help='perceptual loss factor')
parser.add_argument('--g_factor', type=float, default=0.5, help='gan loss factor')
parser.add_argument('--glr', type=float, default=1e-4, help='generator learning rate')
parser.add_argument('--dlr', type=float, default=1e-4, help='discriminator learning rate')
parser.add_argument('--epochs', type=int, default=10000, help='number of epochs to train')
parser.add_argument('--lr_step_size', type=int, default=2000, help='period of learning rate decay')
parser.add_argument('--lr_gamma', type=float, default=0.5, help='multiplicative factor of learning rate decay')
parser.add_argument('--patch_gan', type=int, default=30, help='Path GAN size')
parser.add_argument('--pool_size', type=int, default=50, help='Buffer size for storing generated samples from G')

# misc
parser.add_argument('--period', type=int, default=1, help='period of printing logs')
parser.add_argument('--gpu', type=int, required=True, help='gpu index')

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)


def train(args):
    print(args)

    # net
    netG = Generator()
    netG = netG.cuda()
    netD = Discriminator()
    netD = netD.cuda()

    # loss
    l1_loss = nn.L1Loss().cuda()
    l2_loss = nn.MSELoss().cuda()
    bce_loss = nn.BCELoss().cuda()

    # opt
    optimizerG = optim.Adam(netG.parameters(), lr=args.glr)
    optimizerD = optim.Adam(netD.parameters(), lr=args.dlr)

    # lr
    schedulerG = lr_scheduler.StepLR(optimizerG, args.lr_step_size, args.lr_gamma)
    schedulerD = lr_scheduler.StepLR(optimizerD, args.lr_step_size, args.lr_gamma)

    # utility for saving models, parameters and logs
    save = SaveData(args.save_dir, args.exp, True)
    save.save_params(args)

    # netG, _ = save.load_model(netG)

    dataset = MyDataset(args.data_dir, is_train=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                                             num_workers=int(args.n_threads))

    real_label = Variable(torch.ones([1, 1, args.patch_gan, args.patch_gan], dtype=torch.float)).cuda()
    fake_label = Variable(torch.zeros([1, 1, args.patch_gan, args.patch_gan], dtype=torch.float)).cuda()

    image_pool = ImagePool(args.pool_size)

    vgg = Vgg16(requires_grad=False)
    vgg.cuda()

    for epoch in range(args.epochs):
        print("* Epoch {}/{}".format(epoch + 1, args.epochs))

        schedulerG.step()
        schedulerD.step()

        d_total_real_loss = 0
        d_total_fake_loss = 0
        d_total_loss = 0

        g_total_res_loss = 0
        g_total_per_loss = 0
        g_total_gan_loss = 0
        g_total_loss = 0

        netG.train()
        netD.train()

        for batch, images in tqdm(enumerate(dataloader)):
            input_image, target_image = images
            input_image = Variable(input_image.cuda())
            target_image = Variable(target_image.cuda())
            output_image = netG(input_image)

            # Update D
            netD.requires_grad(True)
            netD.zero_grad()

            ## real image
            real_output = netD(target_image)
            d_real_loss = bce_loss(real_output, real_label)
            d_real_loss.backward()
            d_real_loss = d_real_loss.data.cpu().numpy()
            d_total_real_loss += d_real_loss

            ## fake image
            fake_image = output_image.detach()
            fake_image = Variable(image_pool.query(fake_image.data))
            fake_output = netD(fake_image)
            d_fake_loss = bce_loss(fake_output, fake_label)
            d_fake_loss.backward()
            d_fake_loss = d_fake_loss.data.cpu().numpy()
            d_total_fake_loss += d_fake_loss

            ## loss
            d_total_loss += d_real_loss + d_fake_loss

            optimizerD.step()

            # Update G
            netD.requires_grad(False)
            netG.zero_grad()

            ## reconstruction loss
            g_res_loss = l1_loss(output_image, target_image)
            g_res_loss.backward(retain_graph=True)
            g_res_loss = g_res_loss.data.cpu().numpy()
            g_total_res_loss += g_res_loss

            ## perceptual loss
            g_per_loss = args.p_factor * l2_loss(vgg(output_image), vgg(target_image))
            g_per_loss.backward(retain_graph=True)
            g_per_loss = g_per_loss.data.cpu().numpy()
            g_total_per_loss += g_per_loss

            ## gan loss
            output = netD(output_image)
            g_gan_loss = args.g_factor * bce_loss(output, real_label)
            g_gan_loss.backward()
            g_gan_loss = g_gan_loss.data.cpu().numpy()
            g_total_gan_loss += g_gan_loss

            ## loss
            g_total_loss += g_res_loss + g_per_loss + g_gan_loss

            optimizerG.step()

        d_total_real_loss = d_total_real_loss / (batch + 1)
        d_total_fake_loss = d_total_fake_loss / (batch + 1)
        d_total_loss = d_total_loss / (batch + 1)
        save.add_scalar('D/real', d_total_real_loss, epoch)
        save.add_scalar('D/fake', d_total_fake_loss, epoch)
        save.add_scalar('D/total', d_total_loss, epoch)

        g_total_res_loss = g_total_res_loss / (batch + 1)
        g_total_per_loss = g_total_per_loss / (batch + 1)
        g_total_gan_loss = g_total_gan_loss / (batch + 1)
        g_total_loss = g_total_loss / (batch + 1)
        save.add_scalar('G/res', g_total_res_loss, epoch)
        save.add_scalar('G/per', g_total_per_loss, epoch)
        save.add_scalar('G/gan', g_total_gan_loss, epoch)
        save.add_scalar('G/total', g_total_loss, epoch)

        if epoch % args.period == 0:
            log = "Train d_loss: {:.5f} \t g_loss: {:.5f}".format(d_total_loss, g_total_loss)
            print(log)
            save.save_log(log)
            save.save_model(netG, epoch)


if __name__ == '__main__':
    train(args)
