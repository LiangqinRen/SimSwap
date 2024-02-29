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

from torchvision.utils import save_image


class SimSwapDefense(nn.Module):
    def __init__(self, logger, args):
        super(SimSwapDefense, self).__init__()

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

    def _get_test_pic_path(self) -> tuple[list[str], list[str]]:
        src_imgs_path = [
            f"{self.project_path}/crop_224/zjl.jpg",
            f"{self.project_path}/crop_224/2.jpg",
        ]
        dst_imgs_path = [
            f"{self.project_path}/crop_224/ds.jpg",
            f"{self.project_path}/crop_224/6.jpg",
        ]

        return src_imgs_path, dst_imgs_path

    """def _get_imgs_id(self, imgs_path: list[str]) -> torch.tensor:
        imgs = [
            test_one_image.transformer_Arcface(Image.open(path).convert("RGB"))
            for path in imgs_path
        ]

        img_id = torch.stack(imgs)
        img_transform = transforms.Compose([transforms.CenterCrop(224)])
        img_id = img_transform(img_id)

        return img_id.cuda()"""

    def _load_src_imgs(self, imgs_path: list[str]) -> torch.tensor:
        transformer = transforms.Compose([transforms.ToTensor()])
        imgs = [transformer(Image.open(path).convert("RGB")) for path in imgs_path]

        img_id = torch.stack(imgs)
        img_transform = transforms.Compose([transforms.CenterCrop(224)])
        img_id = img_transform(img_id)

        return img_id.cuda()

    """def _get_imgs_att(self, imgs_path: list[str]) -> torch.tensor:
        imgs = [
            test_one_image.transformer(Image.open(path).convert("RGB"))
            for path in imgs_path
        ]

        img_att = torch.stack(imgs)
        img_transform = transforms.Compose([transforms.CenterCrop(224)])
        img_att = img_transform(img_att)

        return img_att.cuda()"""

    def _load_dst_imgs(self, imgs_path: list[str]) -> torch.tensor:
        transformer = transforms.Compose([transforms.ToTensor()])
        imgs = [transformer(Image.open(path).convert("RGB")) for path in imgs_path]

        img_att = torch.stack(imgs)
        img_transform = transforms.Compose([transforms.CenterCrop(224)])
        img_att = img_transform(img_att)

        return img_att.cuda()

    def _get_img_prior(self, img: torch.tensor) -> torch.tensor:
        img_downsample = F.interpolate(img, size=(112, 112))
        prior = self.target.netArc(img_downsample)
        prior = prior / torch.norm(prior, p=2, dim=1)[0]

        return prior.cuda()

    """def _restore_swap_img(self, swap_img: torch.tensor) -> list[torch.tensor]:
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

        return swap_imgs"""

    def void(self, args):
        save_path = f"../log/{args.ID}/{args.project}_void.png"

        self.target.eval()

        swap_count = 3
        src_imgs, dst_imgs = self._get_random_pic_path(swap_count)
        src_imgs = self._load_src_imgs(src_imgs)
        dst_imgs = self._load_dst_imgs(dst_imgs)
        src_prior = self._get_img_prior(src_imgs)

        swap_img = self.target(None, dst_imgs, src_prior, None, True)

        self.logger.info(f"save the result at log/{args.ID}/{args.project}_void.png")

        results = torch.cat((src_imgs, dst_imgs, swap_img), dim=0)
        save_image(results, save_path, nrow=swap_count)

    def _get_train_pic_path(self, batch_size: int) -> tuple[list[str], list[str]]:
        people = os.listdir(self.dataset_path)
        people1, people2 = random.sample(people, 2)  # "trump", "cage"

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

    def _get_pert_imgs(self, imgs_path: list[str]) -> tuple[torch.tensor, torch.tensor]:
        img_id = self._get_imgs_id(imgs_path)
        pert_imgs = self.GAN_G(img_id)

        return pert_imgs

    def _get_shifted_imgs(self, img: torch.tensor) -> torch.tensor:
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

    """def _save_gan_imgs(
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
        cv2.imwrite(save_path, output)"""

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

    """def _restore_img(
        self, swap_img: torch.tensor, detransform=False
    ) -> list[torch.tensor]:
        swap_imgs = torch.chunk(swap_img, chunks=swap_img.shape[0], dim=0)
        swap_imgs = list(swap_imgs)

        for i in range(len(swap_imgs)):
            if detransform:
                swap_imgs[i] = test_one_image.detransformer(swap_imgs[i])
            swap_imgs[i] = swap_imgs[i].view(
                swap_imgs[i].shape[1], swap_imgs[i].shape[2], swap_imgs[i].shape[3]
            )
            swap_imgs[i] = swap_imgs[i].detach()
            swap_imgs[i] = swap_imgs[i].permute(1, 2, 0)
            swap_imgs[i] = swap_imgs[i].to("cpu")
            swap_imgs[i] = np.array(swap_imgs[i])
            swap_imgs[i] = swap_imgs[i][..., ::-1]
            swap_imgs[i] *= 255
        return swap_imgs"""

    """def _save_imgs(
        self,
        imgs: list[list[torch.tensor]],
        save_path: str,
        save_count: str = 3,
        rows: int = 5,
    ) -> None:
        groups = []
        for i in range(save_count):
            row_groups = []
            for row in range(rows):
                set_of_imgs = []
                for j in range(len(imgs)):
                    img_group = imgs[j]
                    set_of_imgs.append(img_group[i * rows + row])
                group = np.concatenate(set_of_imgs, axis=1)
                row_groups.append(group)
            groups.append(np.concatenate(row_groups, axis=0))

        output = np.concatenate(groups, axis=1)
        cv2.imwrite(save_path, output)"""

    def PGD_SRC(
        self,
        args,
        epsilon=1e-2,
        limit=1e-1,
        loss_ratio=[
            1,
            1,
        ],
        iters=100,
    ):
        save_path = f"../log/{args.ID}/{args.project}_pgd_src.png"

        self.target.to("cuda").eval()
        l2_loss = nn.MSELoss().cuda()

        src_imgs_path = [
            f"{self.project_path}/crop_224/zjl.jpg",
        ]
        tgt_imgs_path = [
            f"{self.project_path}/crop_224/james.jpg",
        ]
        dst_imgs_path = [
            f"{self.project_path}/crop_224/zrf.jpg",
        ]

        src_imgs = self._load_src_imgs(src_imgs_path)
        tgt_imgs = self._load_src_imgs(tgt_imgs_path)
        dst_imgs = self._load_dst_imgs(dst_imgs_path)
        src_prior = self._get_img_prior(src_imgs)
        tgt_prior = self._get_img_prior(tgt_imgs)

        src_swapped_img = self.target(None, dst_imgs, src_prior, None, True)
        tgt_swapped_img = self.target(None, dst_imgs, tgt_prior, None, True)
        raw_results = torch.cat(
            (src_imgs, tgt_imgs, dst_imgs, src_swapped_img, tgt_swapped_img), 0
        )

        x_imgs = src_imgs.clone().detach()
        x_backup = src_imgs.clone().detach()
        epsilon = epsilon * (torch.max(src_imgs) - torch.min(src_imgs)) / 2
        for iter in range(iters):
            x_imgs.requires_grad = True

            x_prior = self._get_img_prior(x_imgs)
            x_swapped_img = self.target(
                None, dst_imgs.detach(), x_prior.detach(), None, True
            )

            swap_diff_loss = l2_loss(x_swapped_img, tgt_swapped_img)
            style_loss = l2_loss(x_prior, tgt_prior.detach())
            loss = loss_ratio[0] * swap_diff_loss + loss_ratio[1] * style_loss
            loss.backward(retain_graph=True)

            x_imgs = x_imgs.clone().detach() + epsilon * x_imgs.grad.sign()
            x_imgs = torch.clamp(x_imgs, min=x_backup - limit, max=x_backup + limit)
            self.logger.info(
                f"[Iter {iter:4}]loss: {loss:.5f}({swap_diff_loss:.5f},{style_loss:.5f})"
            )

        x_prior = self._get_img_prior(x_imgs).detach()
        x_swapped_img = self.target(None, dst_imgs, x_prior, None, True)
        protect_results = torch.cat((x_imgs, x_swapped_img), 0)

        self.logger.info(f"save the result at log/{args.ID}/{args.project}_pgd_src.png")

        results = torch.cat((raw_results, protect_results), dim=0)
        save_image(results, save_path, nrow=5)

    def PGD_DST(
        self,
        args,
        epsilon=3e-3,
        limit=4e-2,
        loss_ratio=[
            1.0,
            0.002,
        ],
        iters=100,
    ):
        save_path = f"../log/{args.ID}/{args.project}_pgd_dst.png"

        self.target.to("cuda").eval()
        l2_loss = nn.MSELoss().cuda()

        src_imgs_path = [
            f"{self.project_path}/crop_224/zjl.jpg",
        ]
        tgt_imgs_path = [
            f"{self.project_path}/crop_224/james.jpg",
        ]
        dst_imgs_path = [
            f"{self.project_path}/crop_224/zrf.jpg",
        ]

        src_imgs = self._load_src_imgs(src_imgs_path)
        tgt_imgs = self._load_src_imgs(tgt_imgs_path)
        dst_imgs = self._load_dst_imgs(dst_imgs_path)
        src_prior = self._get_img_prior(src_imgs)
        tgt_prior = self._get_img_prior(tgt_imgs)

        src_swapped_img = self.target(None, dst_imgs, src_prior, None, True)
        tgt_swapped_img = self.target(None, dst_imgs, tgt_prior, None, True)
        raw_results = torch.cat(
            (src_imgs, tgt_imgs, dst_imgs, src_swapped_img, tgt_swapped_img), 0
        )

        x_imgs = dst_imgs.clone().detach()
        x_backup = dst_imgs.clone().detach()
        epsilon = epsilon * (torch.max(src_imgs) - torch.min(src_imgs)) / 2
        for iter in range(iters):
            self.target.zero_grad()
            x_imgs.requires_grad = True

            x_swapped_img = self.target(None, x_imgs, src_prior.detach(), None, True)
            x_latent_code = self.target.netG.encoder(x_imgs)

            latent_code_diff_loss = -l2_loss(
                x_latent_code, torch.zeros_like(x_latent_code)
            )
            swap_diff_loss = -l2_loss(x_swapped_img, torch.zeros_like(x_swapped_img))

            loss = (
                loss_ratio[0] * swap_diff_loss + loss_ratio[1] * latent_code_diff_loss
            )
            loss.backward()

            x_imgs = x_imgs.clone().detach() + epsilon * x_imgs.grad.sign()
            x_imgs = torch.clamp(x_imgs, min=x_backup - limit, max=x_backup + limit)
            self.logger.info(
                f"[Iter {iter:4}]loss: {loss:.5f}({swap_diff_loss:.5f}, {latent_code_diff_loss:.5f})"
            )

        x_swapped_img = self.target(None, x_imgs, src_prior.detach(), None, True)
        protect_results = torch.cat((x_imgs, x_swapped_img), 0)

        self.logger.info(f"save the result at log/{args.ID}/{args.project}_pgd_dst.png")

        results = torch.cat((raw_results, protect_results), dim=0)
        save_image(results, save_path, nrow=5)

    def GAN_ID(
        self,
        args,
        lr_g=5e-4,
        lr_d=2e-4,
        loss_ratio=[0, 1, 0, 1],
    ):
        save_count = 3

        optimizer_G = optim.Adam(
            [
                {"params": self.GAN_G.up1.parameters()},
                {"params": self.GAN_G.up2.parameters()},
                {"params": self.GAN_G.up3.parameters()},
                {"params": self.GAN_G.up4.parameters()},
                {"params": self.GAN_G.last_layer.parameters()},
            ],
            lr=lr_g,
            betas=(0.5, 0.999),
        )
        optimizer_D = optim.RMSprop(self.GAN_D.parameters(), lr=lr_d)

        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_G, args.epoch)

        self.target.to("cuda").eval()
        self.GAN_G.to("cuda").train()
        self.GAN_D.to("cuda").eval()

        l2_loss = nn.MSELoss().cuda()
        flatten = nn.Flatten().cuda()

        best_loss = float("inf")
        for epoch in range(args.epoch):
            src_imgs, dst_imgs = self._get_train_pic_path(args.batch_size)
            save_path = f"../log/{args.ID}/{args.project}_ID_{epoch}.png"

            img_att = self._get_imgs_att(dst_imgs)
            img_id = self._get_imgs_id(src_imgs)
            latent_id = self._get_img_prior(img_id)
            swap_img = self.target(None, img_att, latent_id, None, True)

            noise, pert_img_id = self.GAN_G(img_id)
            # noise, pert_img_id = self._get_pert_imgs(src_imgs)
            pert_latent_id = self._get_img_prior(pert_img_id)
            pert_swap_img = self.target(None, img_att, pert_latent_id, None, True)

            img_latent_code = self.target.netG.encoder(img_id)
            pert_img_latent_code = self.target.netG.encoder(pert_img_id)

            self.GAN_G.zero_grad()
            defense_diff_loss = l2_loss(flatten(img_id), flatten(pert_img_id))
            swap_diff_loss = -l2_loss(flatten(swap_img), flatten(pert_swap_img))
            latent_code_diff_loss = -l2_loss(img_latent_code, pert_img_latent_code)

            img_id_downsample1 = F.interpolate(img_id, size=(112, 112))
            latent_id1 = self.target.netArc(img_id_downsample1)

            img_id_downsample2 = F.interpolate(pert_img_id, size=(112, 112))
            latent_id2 = self.target.netArc(img_id_downsample2)
            id_loss = -l2_loss(latent_id, pert_latent_id)

            G_loss = (
                loss_ratio[0] * defense_diff_loss
                + loss_ratio[1] * swap_diff_loss
                + loss_ratio[2] * latent_code_diff_loss
                + loss_ratio[3] * id_loss
            )
            G_loss.backward()
            optimizer_G.step()

            self.GAN_G.eval()
            self.GAN_D.train()
            self.GAN_D.zero_grad()

            self.logger.info(
                f"[Epoch {epoch:4}]loss: {G_loss:.5f}({defense_diff_loss:.5f}, {swap_diff_loss:.5f}, {latent_code_diff_loss:.5f}, {id_loss:.5f})"
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
            D_loss = 0
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
            latent_id = self._get_img_prior(img_id)
            swap_img = self.target(img_id, img_att, latent_id, latent_id, True)

            noise, pert_img_id = self._get_pert_imgs(src_imgs)
            pert_latent_id = self._get_img_prior(pert_img_id)
            pert_swap_img = self.target(None, img_att, pert_latent_id, None, True)

            img_latent_code = self.target.netG.encoder(img_id)  # inout att
            pert_img_latent_code = self.target.netG.encoder(pert_img_id)

            shift_img_id = self._get_shifted_imgs(img_id)
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

    def GAN_DST(
        self,
        args,
        lr_g=5e-5,
        loss_ratio=[0.1, 0.9, 0.002, 0.0, 0.0],
        clipped=[0.05, 2.0],
        epochs=1000,
        img_conut=7,
    ):
        self.target.eval()
        self.GAN_G.load_state_dict(self.target.netG.state_dict(), strict=False)

        src_imgs_path, dst_imgs_path = self._get_random_pic_path(img_conut)
        src_imgs = self._load_src_imgs(src_imgs_path)
        dst_imgs = self._load_dst_imgs(dst_imgs_path)
        img_prior = self._get_img_prior(src_imgs)
        swapped_img = self.target(None, dst_imgs, img_prior, None, True)

        # results = torch.cat((src_imgs, dst_imgs, swapped_img), dim=0)
        # save_image(results, "trash.png", nrow=img_conut)
        # quit()

        optimizer_G = optim.Adam(
            [
                {"params": self.GAN_G.up1.parameters()},
                {"params": self.GAN_G.up2.parameters()},
                {"params": self.GAN_G.up3.parameters()},
                {"params": self.GAN_G.up4.parameters()},
                {"params": self.GAN_G.last_layer.parameters()},
            ],
            lr=lr_g,
            betas=(0.5, 0.999),
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_G, epochs)

        self.target.to("cuda").eval()
        self.GAN_G.to("cuda").train()

        l2_loss = nn.MSELoss().cuda()

        for param in self.GAN_G.first_layer.parameters():
            param.requires_grad = False
        for param in self.GAN_G.down1.parameters():
            param.requires_grad = False
        for param in self.GAN_G.down2.parameters():
            param.requires_grad = False
        for param in self.GAN_G.down3.parameters():
            param.requires_grad = False
        for param in self.GAN_G.down4.parameters():
            param.requires_grad = False

        best_loss = float("inf")
        for epoch in range(epochs):
            save_path = f"../log/{args.ID}/{args.project}_gan_dst_{epoch}.png"

            self.GAN_G.train()

            src_imgs_path, dst_imgs_path = self._get_train_pic_path(args.batch_size)

            src_imgs = self._load_src_imgs(src_imgs_path)
            dst_imgs = self._load_dst_imgs(dst_imgs_path)
            img_prior = self._get_img_prior(src_imgs)

            swapped_img = self.target(None, dst_imgs, img_prior, None, True)
            img_latent_code = self.target.netG.encoder(dst_imgs)

            pert_imgs = self.GAN_G(dst_imgs)
            pert_swapped_img = self.target(None, pert_imgs, img_prior, None, True)
            pert_img_latent_code = self.target.netG.encoder(pert_imgs)

            shift_imgs = self._get_shifted_imgs(dst_imgs)
            shift_img_latent_code = self.target.netG.encoder(shift_imgs)

            pert_diff_loss = l2_loss(dst_imgs, pert_imgs)
            swap_diff_loss = -torch.clamp(
                l2_loss(swapped_img, pert_swapped_img), 0.0, clipped[0]
            )
            latent_diff_loss = -l2_loss(img_latent_code, pert_img_latent_code)
            shift_latent_diff_loss = l2_loss(
                shift_img_latent_code, pert_img_latent_code
            )
            self.GAN_G.zero_grad()
            G_loss = (
                loss_ratio[0] * pert_diff_loss
                + loss_ratio[1] * swap_diff_loss
                + loss_ratio[2] * latent_diff_loss
                + loss_ratio[3] * shift_latent_diff_loss
            )

            G_loss.backward()
            optimizer_G.step()
            scheduler.step()

            self.logger.info(
                f"[Epoch {epoch:4}]loss: {G_loss:.5f}({pert_diff_loss:.5f}, {swap_diff_loss:.5f}, {latent_diff_loss:.5f}, {shift_latent_diff_loss:.5f})"
            )

            if epoch % args.save_interval == 0:
                with torch.no_grad():
                    self.GAN_G.eval()
                    self.target.eval()
                    src_imgs_path = [f"{self.project_path}/crop_224/zjl.jpg"]
                    dst_imgs_path = [f"{self.project_path}/crop_224/ds.jpg"]

                    src_imgs = self._load_src_imgs(src_imgs_path)
                    dst_imgs = self._load_dst_imgs(dst_imgs_path)
                    img_prior = self._get_img_prior(src_imgs)

                    swapped_img = self.target(None, dst_imgs, img_prior, None, True)
                    pert_imgs = self.GAN_G(dst_imgs)
                    pert_swapped_img = self.target(
                        None, pert_imgs, img_prior, None, True
                    )

                    results = torch.cat(
                        (src_imgs, dst_imgs, swapped_img, pert_imgs, pert_swapped_img),
                        dim=0,
                    )
                    save_image(results, save_path)

                    """test_protect_img = self.GAN_G(dst_imgs)
                    test_pert_swap_img = self.target(
                        None, test_protect_img, img_prior, None, True
                    )
                    results = torch.cat(
                        (
                            src_imgs[0][None,],
                            dst_imgs[0][None,],
                            test_protect_img[0][None,],
                            swapped_img[0][None,],
                            test_pert_swap_img[0][None,],
                        ),
                        dim=0,
                    )"""
                    save_image(results, save_path)
                    """groups = [
                        test_img_att,
                        test_swap_img,
                        test_protect_img,
                        test_pert_swap_img,
                    ]
                    detrans = [False, False, False, False]
                    save_groups = [
                        self._restore_img(group, detransform=detrans[i])
                        for i, group in enumerate(groups)
                    ]
                    self._save_imgs(save_groups, save_path)"""

            if G_loss.data < best_loss:
                best_loss = G_loss.data
                log_save_path = f"../log/{args.ID}/{args.project}.pth"
                checkpoint_save_path = f"../checkpoint/{args.project}.pth"
                self._save_checkpoint(
                    args, log_save_path, optimizer_G, optimizer_G, G_loss, G_loss
                )
                self._save_checkpoint(
                    args, checkpoint_save_path, optimizer_G, optimizer_G, G_loss, G_loss
                )

    def GAN_clip(
        self,
        args,
        lr_g=5e-5,
        lr_d=2e-4,
        loss_ratio=[0.1, 0.9, 0.002, 0.0, 0.0],
        clipped=[0.05, 2.0],
    ):
        self.GAN_G.load_state_dict(self.target.netG.state_dict(), strict=False)
        self.target.eval()
        test_src_imgs, test_dst_imgs = self._get_random_pic_path(15)
        test_img_id = self._get_imgs_id(test_src_imgs)
        test_img_att = self._get_imgs_att(test_dst_imgs)
        test_latent_id = self._get_img_prior(test_img_id)
        test_swap_img = self.target(None, test_img_att, test_latent_id, None, True)

        # optimizer_G = optim.Adam(self.GAN_G.parameters(), lr=lr_g, betas=(0.5, 0.999))
        # only optimize the parameters of the up1, up2, up3, up4, last_layer in the generator
        optimizer_G = optim.Adam(
            [
                {"params": self.GAN_G.up1.parameters()},
                {"params": self.GAN_G.up2.parameters()},
                {"params": self.GAN_G.up3.parameters()},
                {"params": self.GAN_G.up4.parameters()},
                {"params": self.GAN_G.last_layer.parameters()},
            ],
            lr=lr_g,
            betas=(0.5, 0.999),
        )

        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_G, args.epoch)

        self.target.to("cuda").eval()
        self.GAN_G.to("cuda").train()

        l1_loss = nn.L1Loss().cuda()
        l2_loss = nn.MSELoss().cuda()
        flatten = nn.Flatten().cuda()

        for param in self.GAN_G.first_layer.parameters():
            param.requires_grad = False
        for param in self.GAN_G.down1.parameters():
            param.requires_grad = False
        for param in self.GAN_G.down2.parameters():
            param.requires_grad = False
        for param in self.GAN_G.down3.parameters():
            param.requires_grad = False
        for param in self.GAN_G.down4.parameters():
            param.requires_grad = False

        best_loss = float("inf")
        for epoch in range(args.epoch):
            self.GAN_G.train()

            src_imgs, dst_imgs = self._get_train_pic_path(args.batch_size)
            save_path = f"../log/{args.ID}/{args.project}_gan_{epoch}.png"

            img_att = self._get_imgs_att(dst_imgs)
            img_id = self._get_imgs_id(src_imgs)
            latent_id = self._get_img_prior(img_id)
            swap_img = self.target(img_id, img_att, latent_id, None, True)
            img_latent_code = self.target.netG.encoder(img_att)

            protect_img = self.GAN_G(img_att)
            pert_swap_img = self.target(img_id, protect_img, latent_id, None, True)
            pert_img_latent_code = self.target.netG.encoder(img_att)

            shift_img_id = self._get_shifted_imgs(img_att)
            shift_img_latent_code = self.target.netG.encoder(img_att)

            self.GAN_G.zero_grad()
            defense_diff_loss = l2_loss(flatten(img_att), flatten(protect_img))
            # swap_diff_loss = -l2_loss(flatten(swap_img), flatten(pert_swap_img))
            swap_diff_loss = -torch.clamp(
                l2_loss(flatten(swap_img), flatten(pert_swap_img)), 0.0, clipped[0]
            )
            # latent_code_diff_loss = -l2_loss(img_latent_code, pert_img_latent_code)
            latent_code_diff_loss = -torch.clamp(
                l2_loss(img_latent_code, pert_img_latent_code), 0.0, clipped[1]
            )
            shift_latent_code_diff_loss = l2_loss(
                shift_img_latent_code, pert_img_latent_code
            )

            G_loss = (
                loss_ratio[0] * defense_diff_loss
                + loss_ratio[1] * swap_diff_loss
                + loss_ratio[2] * latent_code_diff_loss
                + loss_ratio[3] * shift_latent_code_diff_loss
            )
            G_loss.backward()
            optimizer_G.step()
            scheduler.step()

            self.logger.info(
                f"[Epoch {epoch:4}]loss: {G_loss:.5f}({defense_diff_loss:.5f}, {swap_diff_loss:.5f}, {latent_code_diff_loss:.5f})"
            )

            if epoch % args.save_interval == 0:
                with torch.no_grad():
                    self.GAN_G.eval()
                    self.target.eval()
                    test_protect_img = self.GAN_G(test_img_att)
                    # print(type(test_protect_img))
                    test_pert_swap_img = self.target(
                        None, test_protect_img, test_latent_id, None, True
                    )
                    groups = [
                        test_img_att,
                        test_swap_img,
                        test_protect_img,
                        test_pert_swap_img,
                    ]
                    detrans = [False, False, False, False]
                    save_groups = [
                        self._restore_img(group, detransform=detrans[i])
                        for i, group in enumerate(groups)
                    ]
                    self._save_imgs(save_groups, save_path)

            if G_loss.data < best_loss:
                best_loss = G_loss.data
                log_save_path = f"../log/{args.ID}/{args.project}.pth"
                checkpoint_save_path = f"../checkpoint/{args.project}.pth"
                self._save_checkpoint(
                    args, log_save_path, optimizer_G, optimizer_G, G_loss, G_loss
                )
                self._save_checkpoint(
                    args, checkpoint_save_path, optimizer_G, optimizer_G, G_loss, G_loss
                )
