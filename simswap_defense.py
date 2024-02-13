import test_one_image

import cv2
import os
import random
import torch
import torchvision

import numpy as np
import torch.nn as nn
import torch.optim as optim
import PIL.Image as Image
import torch.nn.functional as F
import torchvision.transforms as transforms

from torch.autograd import Variable


class SimSwapDefense(nn.Module):
    def __init__(self, logger, args):
        super(SimSwapDefense, self).__init__()

        # all relative paths start with main.py!
        self.project_path = os.path.dirname(os.path.abspath(__file__))
        self.dataset_path = (
            f"{self.project_path}/crop_224/vggface2_crop_arcfacealign_224"
        )
        self.logger = logger
        self.args = args

        from options.test_options import TestOptions
        from models.models import create_model
        from models.fs_networks import Generator, Defense_Discriminator

        self.opt = TestOptions().parse()
        torch.nn.Module.dump_patches = True

        self.target = create_model(self.opt)

        self.GAN_G = Generator(input_nc=3, output_nc=3)
        self.GAN_D = Defense_Discriminator()

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

        return src_select_imgs, dst_select_imgs

    def _get_imgs_id(self, imgs_path: list[str]) -> torch.tensor:
        imgs = [
            test_one_image.transformer_Arcface(Image.open(path).convert("RGB"))
            for path in imgs_path
        ]

        img_id = torch.stack(imgs)
        img_transform = transforms.Compose([transforms.CenterCrop(224)])
        img_id = img_transform(img_id)

        return img_id.cuda()

    def _get_imgs_att(self, imgs_path: list[str]) -> torch.tensor:
        imgs = [
            test_one_image.transformer(Image.open(path).convert("RGB"))
            for path in imgs_path
        ]

        img_att = torch.stack(imgs)
        img_transform = transforms.Compose([transforms.CenterCrop(224)])
        img_att = img_transform(img_att)

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

    def _save_void_imgs(
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

    def void(self, args):
        save_path = f"../log/{args.ID}/{args.project}_void.png"

        self.target.eval()

        swap_count = 3
        src_imgs, dst_imgs = self._get_random_pic_path(swap_count)
        img_id = self._get_imgs_id(src_imgs)
        img_att = self._get_imgs_att(dst_imgs)
        latent_id = self._get_latent_id(img_id)

        swap_img = self.target(img_id, img_att, latent_id, latent_id, True)
        swap_imgs = self._restore_swap_img(swap_img)

        self._save_void_imgs(src_imgs, dst_imgs, swap_imgs, save_path)

    def _get_train_pic_path(self, batch_size: int) -> tuple[list[str], list[str]]:
        people = os.listdir(self.dataset_path)
        people1, people2 = "trump", "cage"  # random.sample(people, 2)

        people1_imgs = [
            f"{self.dataset_path}/{people1}/{i}"
            for i in os.listdir(f"{self.dataset_path}/{people1}")
        ]
        people2_imgs = [
            f"{self.dataset_path}/{people2}/{i}"
            for i in os.listdir(f"{self.dataset_path}/{people2}")
        ]

        min_count = min(batch_size, len(people1_imgs), len(people2_imgs))

        return random.sample(people1_imgs, k=min_count), random.sample(
            people2_imgs, k=min_count
        )

    def _get_pert_imgs_id(
        self, imgs_path: list[str]
    ) -> tuple[torch.tensor, torch.tensor]:
        img_id = self._get_imgs_id(imgs_path)
        noise, pert_imgs = self.GAN_G(img_id)

        return noise, pert_imgs

    def _get_shifted_img_id(self, img: torch.tensor) -> torch.tensor:
        shifted_img = torch.roll(img.clone().detach(), shifts=(-1, 1), dims=(2, 3))
        rotated_img = torchvision.transforms.functional.rotate(shifted_img, 330)
        # scaled_img = torch.nn.functional.interpolate(rotated_img, 54)
        # manipulated_face = F.pad(scaled_img, (5, 5, 5, 5), "constant", 0)

        return rotated_img

    def _center_crop(self, img: np.array, dim: list[int]):
        width, height = img.shape[1], img.shape[0]

        crop_width = dim[0] if dim[0] < img.shape[1] else img.shape[1]
        crop_height = dim[1] if dim[1] < img.shape[0] else img.shape[0]
        mid_x, mid_y = int(width / 2), int(height / 2)
        cw2, ch2 = int(crop_width / 2), int(crop_height / 2)
        crop_img = img[mid_y - ch2 : mid_y + ch2, mid_x - cw2 : mid_x + cw2]
        return crop_img

    def _save_gan_imgs(
        self,
        src_imgs: list[str],
        dst_imgs: list[str],
        swap_imgs: list[torch.tensor],
        noises: list[torch.tensor],
        pert_imgs: list[torch.tensor],
        pert_swap_imgs: list[torch.tensor],
        save_count: int,
        save_path: str,
    ) -> None:
        groups = []
        for i in range(len(src_imgs[:save_count])):
            group = np.concatenate(
                (
                    self._center_crop(cv2.imread(src_imgs[i]), [224, 224]),
                    self._center_crop(cv2.imread(dst_imgs[i]), [224, 224]),
                    # cv2.imread(dst_imgs[i]),
                    swap_imgs[i],
                    noises[i],
                    pert_imgs[i],
                    pert_swap_imgs[i],
                ),
                axis=1,
            )
            groups.append(group)

        output = np.concatenate(groups[:save_count], axis=0)
        cv2.imwrite(save_path, output)

    def _save_checkpoint(
        self,
        args,
        path: str,
        GAN_G_optim: torch.optim,
        GAN_D_optim: torch.optim,
        GAN_G_loss: torch.tensor,
        GAN_D_loss: torch.tensor,
    ) -> None:
        torch.save(
            {
                "epoch": args.epoch,
                "GAN_G_state_dict": self.GAN_G.state_dict(),
                "GAN_D_state_dict": self.GAN_D.state_dict(),
                "GAN_G_optim": GAN_G_optim.state_dict(),
                "GAN_D_optim": GAN_D_optim.state_dict(),
                "GAN_G_loss": GAN_G_loss,
                "GAN_D_loss": GAN_D_loss,
            },
            path,
        )

    def GAN(
        self,
        args,
        lr_g=5e-4,
        lr_d=2e-4,
        loss_ratio=[1000, 1000, 0.0003, 0.0003, 1],
    ):
        save_count = 3

        optimizer_G = optim.Adam(self.GAN_G.parameters(), lr=lr_g, betas=(0.5, 0.999))
        optimizer_D = optim.RMSprop(self.GAN_D.parameters(), lr=lr_d)

        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_G, args.epoch)

        self.target.to("cuda").eval()
        self.GAN_G.to("cuda").train()
        self.GAN_D.to("cuda").eval()

        l1_loss = nn.L1Loss().cuda()
        l2_loss = nn.MSELoss().cuda()
        flatten = nn.Flatten().cuda()

        best_loss = float("inf")
        for epoch in range(args.epoch):
            src_imgs, dst_imgs = self._get_train_pic_path(args.batch_size)
            save_path = f"../log/{args.ID}/{args.project}_gan_{epoch}.png"

            img_att = self._get_imgs_att(dst_imgs)
            img_id = self._get_imgs_id(src_imgs)
            latent_id = self._get_latent_id(img_id)
            swap_img = self.target(img_id, img_att, latent_id, latent_id, True)

            noise, pert_img_id = self._get_pert_imgs_id(src_imgs)
            pert_latent_id = self._get_latent_id(pert_img_id)
            pert_swap_img = self.target(
                pert_img_id, img_att, pert_latent_id, pert_latent_id, True
            )

            img_latent_code = self.target.netG.encoder(img_id)
            pert_img_latent_code = self.target.netG.encoder(pert_img_id)

            shift_img_id = self._get_shifted_img_id(img_id)
            shift_img_latent_code = self.target.netG.encoder(shift_img_id)

            self.GAN_G.zero_grad()
            defense_diff_loss = l2_loss(flatten(img_id), flatten(pert_img_id))
            swap_diff_loss = -l2_loss(flatten(swap_img), flatten(pert_swap_img))
            latent_code_diff_loss = -l2_loss(img_latent_code, pert_img_latent_code)
            shift_latent_code_diff_loss = l2_loss(
                shift_img_latent_code, pert_img_latent_code
            )

            D_loss = 0
            if self.args.use_disc:
                D_loss = self.GAN_D(pert_img_id).mean() - self.GAN_D(img_id).mean()
            G_loss = (
                loss_ratio[0] * defense_diff_loss
                + loss_ratio[1] * swap_diff_loss
                + loss_ratio[2] * latent_code_diff_loss
                + loss_ratio[3] * shift_latent_code_diff_loss
                + loss_ratio[4] * D_loss
            )
            G_loss.backward()
            optimizer_G.step()

            self.GAN_G.eval()
            self.GAN_D.train()
            self.GAN_D.zero_grad()

            if self.args.use_disc:
                D_loss.backward()
                optimizer_D.step()
                scheduler.step()

            self.logger.info(
                f"[Epoch {epoch:4}]loss: {G_loss:.5f}({defense_diff_loss:.5f}, {swap_diff_loss:.5f}, {latent_code_diff_loss:.5f}, {shift_latent_code_diff_loss:.5f}, {D_loss:.5f})"
            )

            swap_imgs = self._restore_swap_img(swap_img)
            noises = self._restore_swap_img(noise)
            pert_imgs = self._restore_swap_img(pert_img_id)
            pert_swap_imgs = self._restore_swap_img(pert_swap_img)

            if epoch % args.save_interval == 0:
                self._save_gan_imgs(
                    src_imgs,
                    dst_imgs,
                    swap_imgs,
                    noises,
                    pert_imgs,
                    pert_swap_imgs,
                    save_count,
                    save_path,
                )

            if G_loss.data < best_loss:
                best_loss = G_loss.data
                log_save_path = f"../log/{args.ID}/{args.project}.pth"
                checkpoint_save_path = f"../checkpoint/{args.project}.pth"
                self._save_checkpoint(
                    args, log_save_path, optimizer_G, optimizer_D, G_loss, D_loss
                )
                self._save_checkpoint(
                    args, checkpoint_save_path, optimizer_G, optimizer_D, G_loss, D_loss
                )
