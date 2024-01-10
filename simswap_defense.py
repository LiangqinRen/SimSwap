import test_one_image

import cv2
import os
import random
import torch

import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
import PIL.Image as Image
import torch.nn.functional as F


class SimSwapDefense(nn.Module):
    def __init__(self, logger):
        super(SimSwapDefense, self).__init__()

        # all relative paths base on main.py!
        self.project_path = os.path.dirname(os.path.abspath(__file__))
        self.logger = logger

        from options.test_options import TestOptions
        from models.models import create_model

        self.opt = TestOptions().parse()
        torch.nn.Module.dump_patches = True
        self.model = create_model(self.opt)

    def _get_random_pic_path(self) -> tuple[str, str]:
        dataset_path = f"{self.project_path}/crop_224/vggface2_crop_arcfacealign_224"

        people = os.listdir(dataset_path)
        people1, people2 = random.sample(people, 2)
        path1 = random.choice(os.listdir(f"{dataset_path}/{people1}"))
        path2 = random.choice(os.listdir(f"{dataset_path}/{people2}"))

        self.logger.debug(f"src: {people1}/{path1}, dst: {people2}/{path2}")

        return f"{dataset_path}/{people1}/{path1}", f"{dataset_path}/{people2}/{path2}"

    def void(self):
        # randomly select two different people and swap their face
        save_path = f"../output/simswap/void.png"

        img_src_path, img_dst_path = self._get_random_pic_path()

        img_src = test_one_image.transformer_Arcface(
            Image.open(img_src_path).convert("RGB")
        )
        img_id = img_src.view(-1, img_src.shape[0], img_src.shape[1], img_src.shape[2])

        img_dst = test_one_image.transformer(Image.open(img_dst_path).convert("RGB"))
        img_att = img_dst.view(-1, img_dst.shape[0], img_dst.shape[1], img_dst.shape[2])

        img_id = img_id.cuda()
        img_att = img_att.cuda()

        self.model.eval()

        img_id_downsample = F.interpolate(img_id, size=(112, 112))
        latent_id = self.model.netArc(img_id_downsample)
        latent_id = latent_id.detach().to("cpu")
        latent_id = latent_id / np.linalg.norm(latent_id, axis=1, keepdims=True)
        latent_id = latent_id.to("cuda")

        img_fake = self.model(img_id, img_att, latent_id, latent_id, True)

        for j in range(img_id.shape[0]):
            if j == 0:
                row1 = img_id[j]
                row2 = img_att[j]
                row3 = img_fake[j]
            else:
                row1 = torch.cat([row1, img_id[j]], dim=2)
                row2 = torch.cat([row2, img_att[j]], dim=2)
                row3 = torch.cat([row3, img_fake[j]], dim=2)

        full = row3.detach()
        full = full.permute(1, 2, 0)
        output = full.to("cpu")
        output = np.array(output)
        output = output[..., ::-1]

        output = output * 255

        output = np.concatenate(
            (cv2.imread(img_src_path), cv2.imread(img_dst_path), output), axis=1
        )
        cv2.imwrite(save_path, output)
