import test_one_image

import cv2
import os
import random
import torch

import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import PIL.Image as Image
import torch.nn.functional as F


class SimSwapDefense(nn.Module):
    def __init__(self, logger):
        super(SimSwapDefense, self).__init__()

        # all relative paths start with main.py!
        self.project_path = os.path.dirname(os.path.abspath(__file__))
        self.dataset_path = (
            f"{self.project_path}/crop_224/vggface2_crop_arcfacealign_224"
        )
        self.logger = logger

        from options.test_options import TestOptions
        from models.models import create_model
        from models.fs_networks import Generator, Discriminator

        self.opt = TestOptions().parse()
        torch.nn.Module.dump_patches = True

        self.target = create_model(self.opt)

        self.GAN_G = Generator(input_nc=3, output_nc=3)
        self.GAN_D = Discriminator(input_nc=3)

    def _get_random_pic_path(self, count: int = 1) -> tuple[list[str], list[str]]:
        people = os.listdir(self.dataset_path)
        src_people, dst_people = random.sample(people, 2)

        src_imgs = os.listdir(f"{self.dataset_path}/{src_people}")
        dst_imgs = os.listdir(f"{self.dataset_path}/{dst_people}")
        count = min(len(src_imgs), len(dst_imgs), count)

        assert count > 0

        src_select_imgs = random.sample(src_imgs, count)
        dst_select_imgs = random.sample(dst_imgs, count)

        src_select_imgs = [
            f"{self.dataset_path}/{src_people}/{img}" for img in src_select_imgs
        ]
        dst_select_imgs = [
            f"{self.dataset_path}/{dst_people}/{img}" for img in dst_select_imgs
        ]

        self.logger.debug(f"src: {src_select_imgs}, dst: {dst_select_imgs}")

        return src_select_imgs, dst_select_imgs

    def _get_imgs_id(self, imgs_path: list[str]) -> torch.tensor:
        imgs = [
            test_one_image.transformer_Arcface(Image.open(path).convert("RGB"))
            for path in imgs_path
        ]

        img_id = torch.stack(imgs)

        return img_id.cuda()

    def _get_imgs_att(self, imgs_path: list[str]) -> torch.tensor:
        imgs = [
            test_one_image.transformer(Image.open(path).convert("RGB"))
            for path in imgs_path
        ]

        img_att = torch.stack(imgs)

        return img_att.cuda()

    def _get_latent_id(self, img_id: torch.tensor) -> torch.tensor:
        img_id_downsample = F.interpolate(img_id, size=(112, 112))
        latent_id = self.target.netArc(img_id_downsample)
        latent_id = latent_id.detach().to("cpu")
        latent_id = latent_id / np.linalg.norm(latent_id, axis=1, keepdims=True)

        return latent_id.cuda()

    def _restore_swap_img(self, swap_img: torch.tensor) -> list[torch.tensor]:
        swap_imgs = torch.chunk(swap_img, chunks=swap_img.shape[0], dim=0)
        swap_imgs = list(swap_imgs)

        for i in range(len(swap_imgs)):
            swap_imgs[i] = swap_imgs[i].view(
                swap_imgs[i].shape[1], swap_imgs[i].shape[2], swap_imgs[i].shape[3]
            )
            swap_imgs[i] = swap_imgs[i].detach()
            swap_imgs[i] = swap_imgs[i].permute(1, 2, 0)
            swap_imgs[i] = swap_imgs[i].to("cpu")
            swap_imgs[i] = np.array(swap_imgs[i])
            swap_imgs[i] = swap_imgs[i][..., ::-1]
            swap_imgs[i] *= 255

        return swap_imgs

    def _save_swap_imgs(
        self,
        src_imgs: list[str],
        dst_imgs: list[str],
        swap_imgs: list[torch.tensor],
        save_path: str,
    ) -> None:
        groups = []
        for i in range(len(src_imgs)):
            group = np.concatenate(
                (cv2.imread(src_imgs[i]), cv2.imread(dst_imgs[i]), swap_imgs[i]), axis=1
            )
            groups.append(group)

        output = np.concatenate(groups, axis=0)
        cv2.imwrite(save_path, output)

    def void(self):
        save_path = f"../output/simswap/void.png"

        self.target.eval()

        swap_count = 3
        src_imgs, dst_imgs = self._get_random_pic_path(swap_count)
        img_id = self._get_imgs_id(src_imgs)
        img_att = self._get_imgs_att(dst_imgs)
        latent_id = self._get_latent_id(img_id)

        swap_img = self.target(img_id, img_att, latent_id, latent_id, True)
        swap_imgs = self._restore_swap_img(swap_img)

        self._save_swap_imgs(src_imgs, dst_imgs, swap_imgs, save_path)

    def _get_train_pic_path(self) -> tuple[list[str], list[str]]:
        people = os.listdir(self.dataset_path)
        people1, people2 = people[100], people[101]  # random.sample(people, 2)

        people1_imgs = [
            f"{self.dataset_path}/{people1}/{i}"
            for i in os.listdir(f"{self.dataset_path}/{people1}")
        ]
        people2_imgs = [
            f"{self.dataset_path}/{people2}/{i}"
            for i in os.listdir(f"{self.dataset_path}/{people2}")
        ]

        min_count = min(len(people1_imgs), len(people2_imgs))

        return people1_imgs[:min_count], people2_imgs[:min_count]

    def GAN(self):
        src_imgs, dst_imgs = self._get_train_pic_path()
        self.logger.debug(
            f"len(src_imgs): {len(src_imgs)}, len(dst_imgs): {len(dst_imgs)}"
        )

        epoch = 64
        optimizer_G = optim.Adam(self.GAN_G.parameters(), lr=5e-5, betas=(0.5, 0.999))
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_G, epoch)

        self.target.to("cuda").eval()
        self.GAN_G.to("cuda").train()

        l1_loss = nn.L1Loss
        l2_loss = nn.MSELoss

        for i in range(64):
            self.logger.info(f"[Epoch {i:4}]")
            for j in range(len(src_imgs)):
                src_img = test_one_image.transformer_Arcface(
                    Image.open(src_imgs[j]).convert("RGB")
                )
                dst_img = test_one_image.transformer(
                    Image.open(dst_imgs[j]).convert("RGB")
                )

                img_id = src_img.view(
                    -1, src_img.shape[0], src_img.shape[1], src_img.shape[2]
                )
                img_att = dst_img.view(
                    -1, dst_img.shape[0], dst_img.shape[1], dst_img.shape[2]
                )

                img_id = img_id.cuda()
                img_att = img_att.cuda()

                img_id_downsample = F.interpolate(img_id, size=(112, 112))
                latent_id = self.target.netArc(img_id_downsample)
                latent_id = latent_id.detach().to("cpu")
                latent_id = latent_id / np.linalg.norm(latent_id, axis=1, keepdims=True)
                latent_id = latent_id.to("cuda")

                # get fake img
                fake_img = self.target(img_id, img_att, latent_id, latent_id, True)

                for k in range(img_id.shape[0]):
                    if k == 0:
                        row1 = img_id[k]
                        row2 = img_att[k]
                        row3 = fake_img[k]
                    else:
                        row1 = torch.cat([row1, img_id[k]], dim=2)
                        row2 = torch.cat([row2, img_att[k]], dim=2)
                        row3 = torch.cat([row3, fake_img[k]], dim=2)

                full = row3.detach()
                full = full.permute(1, 2, 0)
                output = full.to("cpu")
                output = np.array(output)
                output = output[..., ::-1]

                fake_img = output * 255

                # get noise
                noise, pert_img = self.GAN_G(img_id)

                for k in range(img_id.shape[0]):
                    if k == 0:
                        row1 = img_id[k]
                        row2 = img_att[k]
                        row3 = noise[k]
                    else:
                        row1 = torch.cat([row1, img_id[k]], dim=2)
                        row2 = torch.cat([row2, img_att[k]], dim=2)
                        row3 = torch.cat([row3, noise[k]], dim=2)

                full = row3.detach()
                full = full.permute(1, 2, 0)
                output = full.to("cpu")
                output = np.array(output)
                output = output[..., ::-1]

                noise = output * 255

                # get pert_img
                for k in range(img_id.shape[0]):
                    if k == 0:
                        row1 = img_id[k]
                        row2 = img_att[k]
                        row3 = pert_img[k]
                    else:
                        row1 = torch.cat([row1, img_id[k]], dim=2)
                        row2 = torch.cat([row2, img_att[k]], dim=2)
                        row3 = torch.cat([row3, pert_img[k]], dim=2)

                full = row3.detach()
                full = full.permute(1, 2, 0)
                output = full.to("cpu")
                output = np.array(output)
                output = output[..., ::-1]

                pert_img_output = output * 255

                pert_img = pert_img.reshape(3, 224, 224)
                img_id = pert_img.view(
                    -1, pert_img.shape[0], pert_img.shape[1], pert_img.shape[2]
                )

                img_id = img_id.cuda()

                img_id_downsample = F.interpolate(img_id, size=(112, 112))
                latent_id = self.target.netArc(img_id_downsample)
                latent_id = latent_id.detach().to("cpu")
                latent_id = latent_id / np.linalg.norm(latent_id, axis=1, keepdims=True)
                latent_id = latent_id.to("cuda")

                fake_pert_img = self.target(img_id, img_att, latent_id, latent_id, True)

                for k in range(img_id.shape[0]):
                    if k == 0:
                        row1 = img_id[k]
                        row2 = img_att[k]
                        row3 = fake_pert_img[k]
                    else:
                        row1 = torch.cat([row1, img_id[k]], dim=2)
                        row2 = torch.cat([row2, img_att[k]], dim=2)
                        row3 = torch.cat([row3, fake_pert_img[k]], dim=2)

                full = row3.detach()
                full = full.permute(1, 2, 0)
                output = full.to("cpu")
                output = np.array(output)
                output = output[..., ::-1]

                fake_pert_img = output * 255

                output = np.concatenate(
                    (
                        cv2.imread(src_imgs[j]),
                        cv2.imread(dst_imgs[j]),
                        fake_img,
                        noise,
                        pert_img_output,
                        fake_pert_img,
                    ),
                    axis=1,
                )

                if j == 0:
                    cv2.imwrite(f"../output/simswap/gan_{i}_{j}.png", output)
