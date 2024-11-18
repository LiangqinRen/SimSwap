from evaluate import Utility, Effectiveness
from models.models import create_model
from options.test_options import TestOptions

import torch
import torchvision.transforms as transforms
import PIL.Image as Image
import torch.nn.functional as F

from os.path import join
from torchvision.utils import save_image


class Base:
    def __init__(self, args, logger):
        super(Base, self).__init__()
        self.args = args
        self.logger = logger

        self.target = create_model(TestOptions().parse())

        self.utility = Utility()
        self.effectiveness = Effectiveness(args.effectiveness_threshold)

    def _load_imgs(self, imgs_path):  #  -> torch.tensor
        transformer = transforms.Compose([transforms.ToTensor()])
        imgs = [transformer(Image.open(path).convert("RGB")) for path in imgs_path]
        imgs = torch.stack(imgs)

        return imgs.cuda()

    def _get_imgs_identity(self, img):  #  -> torch.tensor
        img_downsample = F.interpolate(img, size=(112, 112))
        prior = self.target.netArc(img_downsample)
        prior = prior / torch.norm(prior, p=2, dim=1)[0]

        return prior.cuda()
