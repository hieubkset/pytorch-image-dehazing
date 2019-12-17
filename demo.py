import argparse
from model import Generator
from PIL import Image
from torch.autograd import Variable
from utils import *

parser = argparse.ArgumentParser(description='image-dehazing')

parser.add_argument('--model', required=True, help='training directory')
parser.add_argument('--images', nargs='+', type=str, default='inputs', help='path to hazy folder')
parser.add_argument('--outdir', default='outputs', help='data save directory')
parser.add_argument('--gpu', type=int, required=True, help='gpu index')

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)


def test(args):
    my_model = Generator()
    my_model.cuda()
    my_model.load_state_dict(torch.load(args.model))
    my_model.eval()

    output_dir = args.outdir
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    img_paths = args.images

    for img_path in img_paths:
        image = Image.open(img_path).convert('RGB')
        width, height = image.size
        scale = 32
        image = image.resize((width // scale * scale, height // scale * scale))
        with torch.no_grad():
            image = rgb_to_tensor(image)
            image = image.unsqueeze(0)
            image = Variable(image.cuda())
            output = my_model(image)
        output = tensor_to_rgb(output)
        out = Image.fromarray(np.uint8(output), mode='RGB')
        out = out.resize((width, height), resample=Image.BICUBIC)
        output_path = os.path.join(output_dir, os.path.basename(img_path))
        out.save(output_path)
        print('One image saved at ' + output_path)


if __name__ == '__main__':
    test(args)
