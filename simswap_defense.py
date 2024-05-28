import os
import random
import time
import math

import torch
import torchvision

import numpy as np
import torch.nn as nn
import torch.optim as optim
import PIL.Image as Image
import torch.nn.functional as F
import torchvision.transforms as transforms
from os.path import join

from torchvision.utils import save_image
from options.test_options import TestOptions
from models.models import create_model
from models.fs_networks import Generator
from tqdm import tqdm

from evaluate import Utility, Efficiency, Evaluate


class SimSwapDefense(nn.Module):
    def __init__(self, args, logger):
        super(SimSwapDefense, self).__init__()
        self.args = args
        self.logger = logger

        self.samples_dir = join(args.data_dir, "samples")
        self.dataset_dir = join(args.data_dir, "vggface2_crop_224")

        self.gan_rgb_limits = [0.075, 0.03, 0.075]
        self.gan_loss_limits = [0.05, 3.0]
        self.gan_src_loss_weights = [1, 1, 0.1]  # pert, swap diff, identity diff
        self.gan_tgt_loss_weights = [
            50,
            10,
            0.02,
            0.005,
        ]  # pert, swap diff, latent diff, rotate latent diff

        self.target = create_model(TestOptions().parse())
        self.GAN_G = Generator(input_nc=3, output_nc=3, epsilon=self.gan_rgb_limits)

        self.utility = Utility()
        self.efficiency = Efficiency(None)

    def _load_imgs(self, imgs_path: list[str]) -> torch.tensor:
        """
        Args:
            imgs_path (list[str]): the path should be the full path
        """
        transformer = transforms.Compose([transforms.ToTensor()])
        imgs = [transformer(Image.open(path).convert("RGB")) for path in imgs_path]
        imgs = torch.stack(imgs)

        return imgs.cuda()

    def _get_imgs_identity(self, img: torch.tensor) -> torch.tensor:
        img_downsample = F.interpolate(img, size=(112, 112))
        prior = self.target.netArc(img_downsample)
        prior = prior / torch.norm(prior, p=2, dim=1)[0]

        return prior.cuda()

    def _save_imgs(self, imgs: list[torch.tensor]) -> None:
        log_dir = self.args.log_dir
        if len(imgs) == 3:
            source, target, swap = imgs[:]
            results = torch.cat((source, target, swap), dim=0)
            save_image(results, join(log_dir, "image", "compare.png"))

            save_image(source, join(log_dir, "image", "source.png"))
            save_image(target, join(log_dir, "image", "target.png"))
            save_image(swap, join(log_dir, "image", "swap.png"))
        elif len(imgs) == 5:  # GAN test
            source, target, swap, pert, pert_swap = imgs[:]
            results = torch.cat((source, target, swap, pert, pert_swap), dim=0)
            save_image(results, join(log_dir, "image", "compare.png"), nrow=3)

            for i, img in enumerate(source, 1):
                save_path = join(log_dir, "image", f"source_{i}.png")
                save_image(img, save_path)
            for i, img in enumerate(target, 1):
                save_path = join(log_dir, "image", f"target_{i}.png")
                save_image(img, save_path)
            for i, img in enumerate(swap, 1):
                save_path = join(log_dir, "image", f"swap_{i}.png")
                save_image(img, save_path)
            for i, img in enumerate(pert, 1):
                save_path = join(log_dir, "image", f"pert_{i}.png")
                save_image(img, save_path)
            for i, img in enumerate(pert_swap, 1):
                save_path = join(log_dir, "image", f"pert_swap_{i}.png")
                save_image(img, save_path)
        elif len(imgs) == 7:
            source, target, swap, mimic, mimic_swap, pert, pert_swap = imgs[:]
            results = torch.cat((source, target, swap, mimic, pert, pert_swap), dim=0)
            save_image(results, join(log_dir, "image", "compare.png"))

            save_image(source, join(log_dir, "image", "source.png"))
            save_image(target, join(log_dir, "image", "target.png"))
            save_image(swap, join(log_dir, "image", "swap.png"))
            save_image(mimic, join(log_dir, "image", "mimic.png"))
            save_image(mimic_swap, join(log_dir, "image", "mimic_swap.png"))
            save_image(pert, join(log_dir, "image", "pert.png"))
            save_image(pert_swap, join(log_dir, "image", "pert_swap.png"))

    def swap(self):
        self.target.eval()

        source_img = self._load_imgs([join(self.args.data_dir, self.args.swap_source)])
        target_img = self._load_imgs([join(self.args.data_dir, self.args.swap_target)])
        source_identity = self._get_imgs_identity(source_img)

        swap_img = self.target(None, target_img, source_identity, None, True)

        self._save_imgs([source_img, target_img, swap_img])

    def calculate_efficiency_threshold(self) -> None:
        dataset_path = join(self.args.data_dir, "vggface2_crop_224")
        all_people = os.listdir(dataset_path)
        imgs_path = []
        for people in all_people:
            people_path = join(dataset_path, people)
            all_imgs_name = os.listdir(people_path)
            selected_imgs_name = random.sample(
                all_imgs_name, min(self.args.metric_people_image, len(all_imgs_name))
            )
            imgs_path.extend([join(people_path, name) for name in selected_imgs_name])

        differences = []
        min_difference, max_difference, sum_difference = 1, 0, 0
        inf = float("inf")
        for path in tqdm(imgs_path):
            source_img = self._load_imgs([path])
            source_identity = self._get_imgs_identity(source_img)
            target_img = self._load_imgs([path])
            swap_img = self.target(None, target_img, source_identity, None, True)

            difference = self.efficiency.get_image_difference(source_img, swap_img)
            if math.isinf(difference):
                continue

            min_difference = min(difference, min_difference)
            max_difference = max(difference, max_difference)
            sum_difference += difference

            differences.append((path, difference))
            tqdm.write(f"{path} difference {difference:.5f}")

        with open(join(self.args.log_dir, "difference.txt"), "w") as f:
            for line in differences:
                f.write(f"{line}\n")

        self.logger.info(
            f"With {len(differences)} pictures, the max, mean, min differences are {max_difference:.5f}, {sum_difference/len(differences):.5f} and {min_difference:.5f}"
        )

    def pgd_source_single(self, loss_weights=[1, 1]):
        self.logger.info(f"loss_weights: {loss_weights}")

        self.target.cuda().eval()
        l2_loss = nn.MSELoss().cuda()

        source_img = self._load_imgs([join(self.args.data_dir, self.args.pgd_source)])
        mimic_img = self._load_imgs([join(self.args.data_dir, self.args.pgd_mimic)])

        source_identity = self._get_imgs_identity(source_img)
        mimic_identity = self._get_imgs_identity(mimic_img)

        x_img = source_img.clone().detach()
        x_backup = source_img.clone().detach()
        epsilon = (
            self.args.pgd_epsilon * (torch.max(source_img) - torch.min(source_img)) / 2
        )
        for iter in range(self.args.pgd_epochs):
            x_img.requires_grad = True

            x_identity = self._get_imgs_identity(x_img)
            pert_diff_loss = l2_loss(x_img, x_backup.detach())
            identity_diff_loss = l2_loss(x_identity, mimic_identity.detach())
            loss = (
                loss_weights[0] * pert_diff_loss + loss_weights[1] * identity_diff_loss
            )

            loss.backward(retain_graph=True)

            x_img = x_img.clone().detach() + epsilon * x_img.grad.sign()
            x_img = torch.clamp(
                x_img,
                min=x_backup - self.args.pgd_limit,
                max=x_backup + self.args.pgd_limit,
            )
            self.logger.info(
                f"[Iter {iter:4}]loss: {loss:.5f}({loss_weights[0] * pert_diff_loss.item():.5f}, {loss_weights[1] * identity_diff_loss.item():.5f})"
            )

        target_img = self._load_imgs([join(self.args.data_dir, self.args.pgd_target)])

        source_swap_img = self.target(None, target_img, source_identity, None, True)
        mimic_swap_img = self.target(None, target_img, mimic_identity, None, True)

        x_identity = self._get_imgs_identity(x_img).detach()
        x_swap_img = self.target(None, target_img, x_identity, None, True)

        self._save_imgs(
            [
                source_img,
                target_img,
                source_swap_img,
                mimic_img,
                mimic_swap_img,
                x_img,
                x_swap_img,
            ]
        )

    def _get_random_imgs_path(self) -> tuple[list[str], list[str]]:
        people = os.listdir(self.dataset_dir)
        random.shuffle(people)

        index = 0
        people_to_select = self.args.pgd_metric_people * 2
        selected_imgs_path = []
        while people_to_select > 0:
            people_to_select -= 1
            imgs_path = os.listdir(join(self.dataset_dir, people[index]))
            while len(imgs_path) < self.args.pgd_people_imgs:
                index += 1
                imgs_path = os.listdir(join(self.dataset_dir, people[index]))

            selected_imgs_name = random.sample(imgs_path, self.args.pgd_people_imgs)
            selected_imgs_path.extend(
                [
                    join(self.dataset_dir, people[index], path)
                    for path in selected_imgs_name
                ]
            )
            index += 1

        source_imgs_path = selected_imgs_path[
            : self.args.pgd_metric_people * self.args.pgd_people_imgs
        ]
        target_imgs_path = selected_imgs_path[
            self.args.pgd_metric_people * self.args.pgd_people_imgs :
        ]

        assert len(source_imgs_path) == len(target_imgs_path)

        return source_imgs_path, target_imgs_path

    def _calculate_utility(
        self, clean_imgs: torch.tensor, pert_imgs: torch.tensor
    ) -> float:
        clean_imgs_ndarray = clean_imgs.detach().cpu().numpy().transpose(0, 2, 3, 1)
        pert_imgs_ndarray = pert_imgs.detach().cpu().numpy().transpose(0, 2, 3, 1)
        ssim = self.utility.compare(clean_imgs_ndarray, pert_imgs_ndarray)

        return ssim

    def _calculate_efficiency(
        self,
        source_imgs: torch.tensor,
        clean_imgs_swap: torch.tensor,
        pert_imgs_swap: torch.tensor,
    ) -> tuple[float, float]:
        source_imgs_ndarray = (
            source_imgs.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255.0
        )
        clean_imgs_swap_ndarray = (
            clean_imgs_swap.detach().cpu().numpy().transpose(0, 2, 3, 1)
        ) * 255.0
        pert_imgs_swap_ndarray = (
            pert_imgs_swap.detach().cpu().numpy().transpose(0, 2, 3, 1)
        ) * 255.0

        source_clean_swap = self.efficiency.compare(
            source_imgs_ndarray, clean_imgs_swap_ndarray
        )
        source_pert_swap = self.efficiency.compare(
            source_imgs_ndarray, pert_imgs_swap_ndarray
        )

        return source_clean_swap, source_pert_swap

    def pgd_source_multiple(self, loss_weights=[1, 1]):
        self.logger.info(f"loss_weights: {loss_weights}")

        self.target.cuda().eval()
        l2_loss = nn.MSELoss().cuda()

        source_imgs_path, target_imgs_path = self._get_random_imgs_path()

        path_index = 0
        batch_size = self.args.pgd_batch_size
        image_dir = join(self.args.log_dir, "image")
        mimic_img = self._load_imgs([join(self.args.data_dir, self.args.pgd_mimic)])
        mimic_identity = self._get_imgs_identity(mimic_img)
        mimic_identity_expand = mimic_identity.detach().expand(batch_size, 512)
        utilities, efficiencies = [], []
        for batch_index in range(self.args.pgd_metric_people):
            src_imgs_path = source_imgs_path[path_index : path_index + batch_size]
            tgt_imgs_path = target_imgs_path[path_index : path_index + batch_size]

            src_imgs = self._load_imgs(src_imgs_path)
            tgt_imgs = self._load_imgs(tgt_imgs_path)

            src_identity = self._get_imgs_identity(src_imgs)

            x_imgs = src_imgs.clone().detach()
            epsilon = (
                self.args.pgd_epsilon * (torch.max(src_imgs) - torch.min(src_imgs)) / 2
            )
            for epoch in range(self.args.pgd_epochs):
                x_imgs.requires_grad = True

                x_identity = torch.empty(batch_size, 512).cuda()
                for i in range(batch_size):
                    identity = self._get_imgs_identity(x_imgs[i].unsqueeze(0))
                    x_identity[i] = identity[0]

                pert_diff_loss = l2_loss(x_imgs, src_imgs.detach())
                identity_diff_loss = l2_loss(x_identity, mimic_identity_expand.detach())

                loss = (
                    loss_weights[0] * pert_diff_loss
                    + loss_weights[1] * identity_diff_loss
                )
                loss.backward()

                x_imgs = x_imgs.clone().detach() + epsilon * x_imgs.grad.sign()
                x_imgs = torch.clamp(
                    x_imgs,
                    min=src_imgs - self.args.pgd_limit,
                    max=src_imgs + self.args.pgd_limit,
                )

                self.logger.info(
                    f"[Batch {batch_index+1:3}/{self.args.pgd_metric_people:3}][Iter {epoch+1:3}/{self.args.pgd_epochs:3}]loss: {loss:.5f}({loss_weights[0] * pert_diff_loss.item():.5f}, {loss_weights[1] * identity_diff_loss.item():.5f})"
                )

            x_identity = self._get_imgs_identity(x_imgs)
            x_swap_img = self.target(None, tgt_imgs, x_identity, None, True)

            src_swap_imgs = self.target(None, tgt_imgs, src_identity, None, True)
            results = torch.cat(
                (src_imgs, tgt_imgs, src_swap_imgs, x_imgs, x_swap_img), dim=0
            )
            save_image(
                results,
                join(image_dir, f"{batch_index+1}_compare.png"),
                nrow=batch_size,
            )

            utility = self._calculate_utility(src_imgs, x_imgs)
            utilities.append(utility)

            source_clean_swap, source_pert_swap = self._calculate_efficiency(
                src_imgs, src_swap_imgs, x_swap_img
            )
            efficiencies.append((source_clean_swap, source_pert_swap))

            self.logger.info(
                f"[Batch {batch_index+1:3}/{self.args.pgd_metric_people:3}]utility: {utility:.3f}, efficiency: {source_clean_swap:.3f}, {source_pert_swap:.3f}"
            )

            path_index += batch_size

        self.logger.info(
            f"Utility: {np.mean(utilities)}, Efficiency: {sum(v[0] for v in efficiencies) / len(efficiencies):.5f},{sum(v[1] for v in efficiencies) / len(efficiencies):.5f}"
        )

    def pgd_target_single(self, loss_weights=[100, 1]) -> None:
        self.logger.info(f"loss_weights: {loss_weights}")

        self.target.cuda().eval()
        l2_loss = nn.MSELoss().cuda()

        source_img = self._load_imgs([join(self.args.data_dir, self.args.pgd_source)])
        mimic_img = self._load_imgs([join(self.args.data_dir, self.args.pgd_mimic)])
        target_img = self._load_imgs([join(self.args.data_dir, self.args.pgd_target)])

        source_identity = self._get_imgs_identity(source_img)

        x_img = target_img.clone().detach()
        epsilon = (
            self.args.pgd_epsilon * (torch.max(target_img) - torch.min(target_img)) / 2
        )
        for iter in range(self.args.pgd_epochs):
            x_img.requires_grad = True

            x_latent_code = self.target.netG.encoder(x_img)
            mimic_latent_code = self.target.netG.encoder(mimic_img)
            pert_diff_loss = l2_loss(x_img, target_img.detach())
            latent_code_diff_loss = l2_loss(x_latent_code, mimic_latent_code.detach())
            loss = (
                loss_weights[0] * pert_diff_loss
                + loss_weights[1] * latent_code_diff_loss
            )

            loss.backward(retain_graph=True)

            x_img = x_img.clone().detach() + epsilon * x_img.grad.sign()
            x_img = torch.clamp(
                x_img,
                min=target_img - self.args.pgd_limit,
                max=target_img + self.args.pgd_limit,
            )
            self.logger.info(
                f"[Iter {iter:4}]loss: {loss:.5f}({loss_weights[0] * pert_diff_loss.item():.5f}, {loss_weights[1] * latent_code_diff_loss.item():.5f})"
            )

        source_swap_img = self.target(None, target_img, source_identity, None, True)
        mimic_swap_img = self.target(None, mimic_img, source_identity, None, True)

        x_swap_img = self.target(None, x_img, source_identity, None, True)

        self._save_imgs(
            [
                source_img,
                target_img,
                source_swap_img,
                mimic_img,
                mimic_swap_img,
                x_img,
                x_swap_img,
            ]
        )

    def pgd_target_multiple(self, loss_weights=[100, 1]) -> None:
        self.logger.info(f"loss_weights: {loss_weights}")

        self.target.cuda().eval()
        l2_loss = nn.MSELoss().cuda()

        source_imgs_path, target_imgs_path = self._get_random_imgs_path()

        path_index = 0
        batch_size = self.args.pgd_batch_size
        image_dir = join(self.args.log_dir, "image")

        mimic_img = self._load_imgs([join(self.args.data_dir, self.args.pgd_mimic)])
        mimic_latent_code = self.target.netG.encoder(mimic_img)
        utilities, efficiencies = [], []
        for batch_index in range(self.args.pgd_metric_people):
            src_imgs_path = source_imgs_path[path_index : path_index + batch_size]
            tgt_imgs_path = target_imgs_path[path_index : path_index + batch_size]

            src_imgs = self._load_imgs(src_imgs_path)
            tgt_imgs = self._load_imgs(tgt_imgs_path)

            x_imgs = tgt_imgs.clone().detach()
            epsilon = (
                self.args.pgd_epsilon * (torch.max(tgt_imgs) - torch.min(tgt_imgs)) / 2
            )
            mimic_latent_code_expand = mimic_latent_code.detach().expand(
                batch_size, 512, 28, 28
            )
            for epoch in range(self.args.pgd_epochs):
                x_imgs.requires_grad = True

                x_latent_code = self.target.netG.encoder(x_imgs)
                pert_diff_loss = l2_loss(x_imgs, tgt_imgs.detach())
                latent_code_diff_loss = l2_loss(
                    x_latent_code, mimic_latent_code_expand.detach()
                )
                loss = (
                    loss_weights[0] * pert_diff_loss
                    + loss_weights[1] * latent_code_diff_loss
                )

                loss.backward(retain_graph=True)

                x_imgs = x_imgs.clone().detach() + epsilon * x_imgs.grad.sign()
                x_imgs = torch.clamp(
                    x_imgs,
                    min=tgt_imgs - self.args.pgd_limit,
                    max=tgt_imgs + self.args.pgd_limit,
                )

                self.logger.info(
                    f"[Batch {batch_index+1:3}/{self.args.pgd_metric_people:3}][Iter {epoch+1:3}/{self.args.pgd_epochs:3}]loss: {loss:.5f}({loss_weights[0] * pert_diff_loss.item():.5f}, {loss_weights[1] * latent_code_diff_loss.item():.5f})"
                )

            source_identity = self._get_imgs_identity(src_imgs)
            source_swap_img = self.target(None, tgt_imgs, source_identity, None, True)

            x_swap_img = self.target(None, x_imgs, source_identity, None, True)
            results = torch.cat(
                (src_imgs, tgt_imgs, source_swap_img, x_imgs, x_swap_img), dim=0
            )
            save_image(
                results,
                join(image_dir, f"{batch_index+1}_compare.png"),
                nrow=batch_size,
            )

            utility = self._calculate_utility(tgt_imgs, x_imgs)
            utilities.append(utility)

            source_clean_swap, source_pert_swap = self._calculate_efficiency(
                src_imgs, source_swap_img, x_swap_img
            )
            efficiencies.append((source_clean_swap, source_pert_swap))

            self.logger.info(
                f"[Batch {batch_index+1:3}/{self.args.pgd_metric_people:3}]utility: {utility:.3f}, efficiency: {source_clean_swap:.3f}, {source_pert_swap:.3f}"
            )
            path_index += batch_size

        self.logger.info(
            f"Utility: {np.mean(utilities)}, Efficiency: {sum(v[0] for v in efficiencies) / len(efficiencies):.5f},{sum(v[1] for v in efficiencies) / len(efficiencies):.5f}"
        )

    def _get_all_imgs_path(self) -> list[str]:
        all_people = os.listdir(self.dataset_dir)
        all_imgs_path = []
        for people in all_people:
            people_dir = join(self.dataset_dir, people)
            all_imgs_name = os.listdir(people_dir)
            all_imgs_path.extend(
                [join(self.dataset_dir, people, name) for name in all_imgs_name]
            )

        self.logger.info(f"Collect {len(all_imgs_path)} images for GAN training")
        return all_imgs_path

    def GAN_SRC(self):
        self.logger.info(
            f"rgb_limits: {self.gan_rgb_limits}, loss_weights: {self.gan_src_loss_weights}"
        )

        self.GAN_G.load_state_dict(self.target.netG.state_dict(), strict=False)
        optimizer_G = optim.Adam(
            [
                {"params": self.GAN_G.up1.parameters()},
                {"params": self.GAN_G.up2.parameters()},
                {"params": self.GAN_G.up3.parameters()},
                {"params": self.GAN_G.up4.parameters()},
                {"params": self.GAN_G.last_layer.parameters()},
            ],
            lr=self.args.gan_generator_lr,
            betas=(0.5, 0.999),
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer_G, self.args.gan_epochs
        )

        self.target.cuda().eval()
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

        checkpoint_dir = join(self.args.log_dir, "checkpoint")
        os.mkdir(checkpoint_dir)

        best_loss = float("inf")
        all_imgs_path = self._get_all_imgs_path()

        mimic_identity_expand = torch.ones(self.args.gan_batch_size, 512).cuda()

        for epoch in range(self.args.gan_epochs):
            self.GAN_G.cuda().train()
            src_imgs_path = random.sample(all_imgs_path, self.args.gan_batch_size)
            tgt_imgs_path = random.sample(all_imgs_path, self.args.gan_batch_size)

            src_imgs = self._load_imgs(src_imgs_path)
            src_identity = self._get_imgs_identity(src_imgs)
            tgt_imgs = self._load_imgs(tgt_imgs_path)

            pert_src_imgs = self.GAN_G(src_imgs)
            pert_src_identity = self._get_imgs_identity(pert_src_imgs)
            pert_swap_imgs = self.target(
                None, tgt_imgs.detach(), pert_src_identity, None, True
            )

            mimic_swap_imgs = self.target(
                None,
                tgt_imgs.detach(),
                mimic_identity_expand.detach(),
                None,
                True,
            )

            self.GAN_G.zero_grad()
            pert_diff_loss = l2_loss(src_imgs, pert_src_imgs)
            swap_diff_loss = l2_loss(pert_swap_imgs, mimic_swap_imgs.detach())
            identity_diff_loss = l2_loss(src_identity, mimic_identity_expand.detach())

            G_loss = (
                self.gan_src_loss_weights[0] * pert_diff_loss
                + self.gan_src_loss_weights[1] * swap_diff_loss
                + self.gan_src_loss_weights[2] * identity_diff_loss
            )
            G_loss.backward()
            optimizer_G.step()
            scheduler.step()

            self.logger.info(
                f"[Epoch {epoch:6}]loss: {G_loss:.5f}({self.gan_src_loss_weights[0] * pert_diff_loss.item():.5f}, {self.gan_src_loss_weights[1] * swap_diff_loss.item():.5f}, {self.gan_src_loss_weights[2] * identity_diff_loss.item():.5f})"
            )

            if epoch % self.args.gan_generator_interval == 0:
                with torch.no_grad():
                    self.GAN_G.eval()
                    self.target.eval()

                    src_imgs_path = random.sample(all_imgs_path, 7)
                    src_imgs_path.extend(
                        [
                            join(self.samples_dir, "zjl.jpg"),
                            join(self.samples_dir, "6.jpg"),
                            join(self.samples_dir, "hzxc.jpg"),
                        ]
                    )
                    tgt_imgs_path = random.sample(all_imgs_path, 7)
                    tgt_imgs_path.extend(
                        [
                            join(self.samples_dir, "zrf.jpg"),
                            join(self.samples_dir, "zrf.jpg"),
                            join(self.samples_dir, "zrf.jpg"),
                        ]
                    )

                    src_imgs = self._load_imgs(src_imgs_path)
                    src_identity = self._get_imgs_identity(src_imgs)
                    tgt_imgs = self._load_imgs(tgt_imgs_path)

                    src_imgs = self._load_imgs(src_imgs_path)
                    tgt_imgs = self._load_imgs(tgt_imgs_path)
                    src_identity = self._get_imgs_identity(src_imgs)

                    src_swap_img = self.target(None, tgt_imgs, src_identity, None, True)
                    mimic_identity_test = torch.ones(10, 512).cuda()
                    mimic_swap_imgs = self.target(
                        None,
                        tgt_imgs,
                        mimic_identity_test,
                        None,
                        True,
                    )
                    raw_results = torch.cat(
                        (src_imgs, tgt_imgs, src_swap_img, mimic_swap_imgs), 0
                    )

                    x_imgs = self.GAN_G(src_imgs)
                    x_identity = self._get_imgs_identity(x_imgs)
                    x_swap_imgs = self.target(None, tgt_imgs, x_identity, None, True)
                    protect_results = torch.cat((x_imgs, x_swap_imgs), 0)

                    save_path = join(self.args.log_dir, "image", f"gan_src_{epoch}.png")
                    self.logger.info(f"save the result at {save_path}")

                    results = torch.cat((raw_results, protect_results), dim=0)
                    save_image(results, save_path, nrow=10)

            if G_loss.data < best_loss:
                best_loss = G_loss.data
                log_save_path = join(self.args.log_dir, "checkpoint", "gan_src.pth")
                torch.save(
                    {
                        "epoch": epoch,
                        "GAN_G_state_dict": self.GAN_G.state_dict(),
                        "GAN_G_loss": G_loss,
                    },
                    log_save_path,
                )

    def _get_shifted_imgs(self, img: torch.tensor) -> torch.tensor:
        shifted_img = torch.roll(img.clone().detach(), shifts=(-1, 1), dims=(2, 3))
        rotated_img = torchvision.transforms.functional.rotate(shifted_img, 330)

        return rotated_img

    def GAN_TGT(self):
        self.logger.info(
            f"rgb_limits: {self.gan_rgb_limits}, loss_limits: {self.gan_loss_limits}, loss_weights: {self.gan_src_loss_weights}"
        )

        self.GAN_G.load_state_dict(self.target.netG.state_dict(), strict=False)
        optimizer_G = optim.Adam(
            [
                {"params": self.GAN_G.up1.parameters()},
                {"params": self.GAN_G.up2.parameters()},
                {"params": self.GAN_G.up3.parameters()},
                {"params": self.GAN_G.up4.parameters()},
                {"params": self.GAN_G.last_layer.parameters()},
            ],
            lr=self.args.gan_generator_lr,
            betas=(0.5, 0.999),
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer_G, self.args.gan_epochs
        )

        self.target.cuda().eval()
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

        checkpoint_dir = join(self.args.log_dir, "checkpoint")
        os.mkdir(checkpoint_dir)

        best_loss = float("inf")
        all_imgs_path = self._get_all_imgs_path()

        for epoch in range(self.args.gan_epochs):
            self.GAN_G.cuda().train()
            src_imgs_path = random.sample(all_imgs_path, self.args.gan_batch_size)
            tgt_imgs_path = random.sample(all_imgs_path, self.args.gan_batch_size)

            src_imgs = self._load_imgs(src_imgs_path)
            src_identity = self._get_imgs_identity(src_imgs)
            tgt_imgs = self._load_imgs(tgt_imgs_path)
            tgt_swap_imgs = self.target(None, tgt_imgs, src_identity, None, True)
            tgt_latent_code = self.target.netG.encoder(tgt_swap_imgs)

            pert_tgt_imgs = self.GAN_G(tgt_imgs)
            pert_swap_imgs = self.target(None, pert_tgt_imgs, src_identity, None, True)
            pert_latent_code = self.target.netG.encoder(pert_tgt_imgs)

            rotate_tgt_imgs = self._get_shifted_imgs(tgt_imgs)
            rotate_latent_code = self.target.netG.encoder(rotate_tgt_imgs)
            # pert, swap diff, latent diff, rotate latent diff
            self.GAN_G.zero_grad()
            pert_diff_loss = l2_loss(flatten(pert_tgt_imgs), flatten(tgt_imgs))
            swap_diff_loss = -torch.clamp(
                l2_loss(flatten(pert_swap_imgs), flatten(tgt_swap_imgs)),
                0.0,
                self.gan_loss_limits[0],
            )
            latent_diff_loss = -torch.clamp(
                l2_loss(flatten(pert_latent_code), flatten(tgt_latent_code)),
                0.0,
                self.gan_loss_limits[1],
            )
            rotate_latent_diff_loss = l2_loss(
                flatten(pert_latent_code), flatten(rotate_latent_code)
            )

            G_loss = (
                self.gan_tgt_loss_weights[0] * pert_diff_loss
                + self.gan_tgt_loss_weights[1] * swap_diff_loss
                + self.gan_tgt_loss_weights[2] * latent_diff_loss
                + self.gan_tgt_loss_weights[3] * rotate_latent_diff_loss
            )
            G_loss.backward()
            optimizer_G.step()
            scheduler.step()

            self.logger.info(
                f"[Epoch {epoch:6}]loss: {G_loss:.5f}({self.gan_tgt_loss_weights[0] * pert_diff_loss.item():.5f}, {self.gan_tgt_loss_weights[1] * swap_diff_loss.item():.5f}, {self.gan_tgt_loss_weights[2] * latent_diff_loss.item():.5f}, {self.gan_tgt_loss_weights[3] * rotate_latent_diff_loss.item():.5f})({swap_diff_loss.item():.5f}, {latent_diff_loss.item():.5f})"
            )

            if epoch % self.args.gan_generator_interval == 0:
                with torch.no_grad():
                    self.GAN_G.eval()
                    self.target.eval()

                    src_imgs_path = random.sample(all_imgs_path, 7)
                    src_imgs_path.extend(
                        [
                            join(self.samples_dir, "zrf.jpg"),
                            join(self.samples_dir, "zrf.jpg"),
                            join(self.samples_dir, "zrf.jpg"),
                        ]
                    )
                    tgt_imgs_path = random.sample(all_imgs_path, 7)
                    tgt_imgs_path.extend(
                        [
                            join(self.samples_dir, "zjl.jpg"),
                            join(self.samples_dir, "6.jpg"),
                            join(self.samples_dir, "hzxc.jpg"),
                        ]
                    )

                    src_imgs = self._load_imgs(src_imgs_path)
                    src_identity = self._get_imgs_identity(src_imgs)
                    tgt_imgs = self._load_imgs(tgt_imgs_path)
                    tgt_swap_img = self.target(None, tgt_imgs, src_identity, None, True)

                    raw_results = torch.cat((src_imgs, tgt_imgs, tgt_swap_img), 0)

                    x_imgs = self.GAN_G(tgt_imgs)
                    x_swap_imgs = self.target(None, x_imgs, src_identity, None, True)
                    protect_results = torch.cat((x_imgs, x_swap_imgs), 0)

                    save_path = join(self.args.log_dir, "image", f"gan_tgt_{epoch}.png")
                    self.logger.info(f"save the result at {save_path}")

                    results = torch.cat((raw_results, protect_results), dim=0)
                    save_image(results, save_path, nrow=10)

            if G_loss.data < best_loss:
                best_loss = G_loss.data
                log_save_path = join(self.args.log_dir, "checkpoint", "gan_tgt.pth")
                torch.save(
                    {
                        "epoch": epoch,
                        "GAN_G_state_dict": self.GAN_G.state_dict(),
                        "GAN_G_loss": G_loss,
                    },
                    log_save_path,
                )

    def _get_split_all_imgs_path(self) -> tuple[list[str], list[str]]:
        all_people = os.listdir(self.dataset_dir)
        random.shuffle(all_people)

        source_people = all_people[: int(len(all_people) / 2)]
        target_people = all_people[int(len(all_people) / 2) :]

        source_imgs_path = []
        for people in source_people:
            people_dir = join(self.dataset_dir, people)
            people_imgs_name = os.listdir(people_dir)
            source_imgs_path.extend(
                [join(self.dataset_dir, people, name) for name in people_imgs_name]
            )

        target_imgs_path = []
        for people in target_people:
            people_dir = join(self.dataset_dir, people)
            people_imgs_name = os.listdir(people_dir)
            target_imgs_path.extend(
                [join(self.dataset_dir, people, name) for name in people_imgs_name]
            )

        return source_imgs_path, target_imgs_path

    def GAN_SRC_test(self):
        from utils import calculate_score

        model_path = join("checkpoints", "gan_src.pth")
        self.GAN_G.load_state_dict(torch.load(model_path)["GAN_G_state_dict"])

        self.target.cuda().eval()
        self.GAN_G.cuda().eval()

        source_path = [
            join(self.samples_dir, "zjl.jpg"),
            join(self.samples_dir, "6.jpg"),
            join(self.samples_dir, "jl.jpg"),
        ]
        target_path = [
            join(self.samples_dir, "zrf.jpg"),
            join(self.samples_dir, "zrf.jpg"),
            join(self.samples_dir, "zrf.jpg"),
        ]

        source_imgs = self._load_imgs(source_path)
        target_imgs = self._load_imgs(target_path)
        source_identity = self._get_imgs_identity(source_imgs)
        swap_imgs = self.target(None, target_imgs, source_identity, None, True)

        pert_source_imgs = self.GAN_G(source_imgs)
        pert_source_identity = self._get_imgs_identity(pert_source_imgs)
        pert_swap_imgs = self.target(
            None, target_imgs, pert_source_identity, None, True
        )

        self._save_imgs(
            [source_imgs, target_imgs, swap_imgs, pert_source_imgs, pert_swap_imgs]
        )

        source_imgs_path, target_imgs_path = self._get_split_all_imgs_path()
        utilities, clean_efficiencies, pert_efficiencies = [], [], []
        scores = []
        for i in range(self.args.gan_test_times):
            iter_source_path = random.sample(source_imgs_path, self.args.gan_batch_size)
            iter_target_path = random.sample(target_imgs_path, self.args.gan_batch_size)

            source_imgs = self._load_imgs(iter_source_path)
            target_imgs = self._load_imgs(iter_target_path)
            source_identity = self._get_img_identity(source_imgs)
            swap_imgs = self.target(None, target_imgs, source_identity, None, True)

            pert_source_imgs = self.GAN_G(source_imgs)
            pert_source_identity = self._get_img_identity(pert_source_imgs)
            pert_swap_imgs = self.target(
                None, target_imgs, pert_source_identity, None, True
            )

            utility = self._calculate_utility(source_imgs, pert_source_imgs)
            utilities.append(utility)
            source_clean_swap, source_pert_swap = self._calculate_efficiency(
                source_imgs, swap_imgs, pert_swap_imgs
            )
            clean_efficiencies.append(source_clean_swap)
            pert_efficiencies.append(source_pert_swap)
            score = calculate_score(utility, source_clean_swap, source_pert_swap)
            scores.append(score)

            self.logger.info(
                f"Iter {i:3}, utility: {utility:.3f}, efficiency: {source_clean_swap:.3f}, {source_pert_swap:.3f}, score: {score:.3f}"
            )

        self.logger.info(
            f"Average of {self.args.gan_batch_size * self.args.gan_test_times} pictures: utility: {sum(utilities)/len(utilities):.3f}, efficiency: {sum(clean_efficiencies)/len(clean_efficiencies):.3f}, {sum(pert_efficiencies)/len(pert_efficiencies):.3f}, score: {sum(scores)/len(scores):.3f}"
        )

    def GAN_TGT_test(self):
        from utils import calculate_score

        model_path = join("checkpoints", "gan_tgt.pth")
        self.GAN_G.load_state_dict(torch.load(model_path)["GAN_G_state_dict"])

        self.target.cuda().eval()
        self.GAN_G.cuda().eval()

        source_path = [
            join(self.samples_dir, "zrf.jpg"),
            join(self.samples_dir, "zrf.jpg"),
            join(self.samples_dir, "zrf.jpg"),
        ]
        target_path = [
            join(self.samples_dir, "zjl.jpg"),
            join(self.samples_dir, "6.jpg"),
            join(self.samples_dir, "jl.jpg"),
        ]

        source_imgs = self._load_imgs(source_path)
        target_imgs = self._load_imgs(target_path)
        source_identity = self._get_imgs_identity(source_imgs)
        swap_imgs = self.target(None, target_imgs, source_identity, None, True)

        pert_target_imgs = self.GAN_G(target_imgs)
        pert_swap_imgs = self.target(
            None, pert_target_imgs, source_identity, None, True
        )

        self._save_imgs(
            [source_imgs, target_imgs, swap_imgs, pert_target_imgs, pert_swap_imgs]
        )

        source_imgs_path, target_imgs_path = self._get_split_all_imgs_path()
        utilities, clean_efficiencies, pert_efficiencies = [], [], []
        scores = []
        for i in range(self.args.gan_test_times):
            iter_source_path = random.sample(source_imgs_path, self.args.gan_batch_size)
            iter_target_path = random.sample(target_imgs_path, self.args.gan_batch_size)

            source_imgs = self._load_imgs(iter_source_path)
            target_imgs = self._load_imgs(iter_target_path)
            source_identity = self._get_imgs_identity(source_imgs)
            swap_imgs = self.target(None, target_imgs, source_identity, None, True)

            pert_target_imgs = self.GAN_G(target_imgs)
            pert_swap_imgs = self.target(
                None, pert_target_imgs, source_identity, None, True
            )

            utility = self._calculate_utility(target_imgs, pert_target_imgs)
            utilities.append(utility)
            source_clean_swap, source_pert_swap = self._calculate_efficiency(
                source_imgs, swap_imgs, pert_swap_imgs
            )
            clean_efficiencies.append(source_clean_swap)
            pert_efficiencies.append(source_pert_swap)
            score = calculate_score(utility, source_clean_swap, source_pert_swap)
            scores.append(score)

            self.logger.info(
                f"Iter {i:3}, utility: {utility:.3f}, efficiency: {source_clean_swap:.3f}, {source_pert_swap:.3f}, score: {score:.3f}"
            )

        self.logger.info(
            f"Average of {self.args.gan_batch_size * self.args.gan_test_times} pictures: utility: {sum(utilities)/len(utilities):.3f}, efficiency: {sum(clean_efficiencies)/len(clean_efficiencies):.3f}, {sum(pert_efficiencies)/len(pert_efficiencies):.3f}, score: {sum(scores)/len(scores):.3f}"
        )


class SimSwapDefensee(nn.Module):
    def __init__(self, args, logger):
        super(SimSwapDefensee, self).__init__()

        self.project_path = os.path.dirname(os.path.abspath(__file__))
        self.dataset_path = (
            f"{self.project_path}/crop_224/vggface2_crop_arcfacealign_224"
        )
        self.args = args
        self.logger = logger

        from options.test_options import TestOptions
        from models.models import create_model
        from models.fs_networks import Generator, Defense_Discriminator

        self.opt = TestOptions().parse()
        torch.nn.Module.dump_patches = True

        self.target = create_model(self.opt)

        self.GAN_G = Generator(input_nc=3, output_nc=3)
        self.GAN_D = Defense_Discriminator()

    def _get_random_img_path(self, batch_size: int = 1) -> tuple[list[str], list[str]]:
        people = os.listdir(self.dataset_path)
        src_people, dst_people = random.sample(people, 2)

        src_imgs = os.listdir(f"{self.dataset_path}/{src_people}")
        dst_imgs = os.listdir(f"{self.dataset_path}/{dst_people}")
        batch_size = min(len(src_imgs), len(dst_imgs), batch_size)

        assert batch_size > 0

        src_select_imgs = random.sample(src_imgs, batch_size)
        dst_select_imgs = random.sample(dst_imgs, batch_size)

        src_select_imgs = [
            f"{self.dataset_path}/{src_people}/{img}" for img in src_select_imgs
        ]
        dst_select_imgs = [
            f"{self.dataset_path}/{dst_people}/{img}" for img in dst_select_imgs
        ]

        return src_select_imgs, dst_select_imgs

    def _load_img(self, img_path: str) -> torch.tensor:
        img_path = os.path.join(self.args.data_dir, img_path)
        transformer = transforms.Compose([transforms.ToTensor()])

        img = [transformer(Image.open(img_path).convert("RGB"))]
        img = torch.stack(img)

        return img.cuda()

    def _load_imgs(self, imgs_path: list[str]) -> torch.tensor:
        # the path in imgs_path should be the full path
        transformer = transforms.Compose([transforms.ToTensor()])
        imgs = [transformer(Image.open(path).convert("RGB")) for path in imgs_path]
        imgs = torch.stack(imgs)

        return imgs.cuda()

    def _load_src_imgs(self, imgs_path: list[str]) -> torch.tensor:
        transformer = transforms.Compose([transforms.ToTensor()])
        imgs = [transformer(Image.open(path).convert("RGB")) for path in imgs_path]
        img_id = torch.stack(imgs)
        img_transform = transforms.Compose([transforms.CenterCrop(224)])
        img_id = img_transform(img_id)

        return img_id.cuda()

    def _load_dst_imgs(self, imgs_path: list[str]) -> torch.tensor:
        transformer = transforms.Compose([transforms.ToTensor()])
        imgs = [transformer(Image.open(path).convert("RGB")) for path in imgs_path]

        img_att = torch.stack(imgs)
        img_transform = transforms.Compose([transforms.CenterCrop(224)])
        img_att = img_transform(img_att)

        return img_att.cuda()

    def _get_img_identity(self, img: torch.tensor) -> torch.tensor:
        img_downsample = F.interpolate(img, size=(112, 112))
        prior = self.target.netArc(img_downsample)
        prior = prior / torch.norm(prior, p=2, dim=1)[0]

        return prior.cuda()

    def swap(self):
        self.target.eval()

        source_img = self._load_img(self.args.swap_source)
        target_img = self._load_img(self.args.swap_target)
        source_identity = self._get_img_identity(source_img)

        swap_img = self.target(None, target_img, source_identity, None, True)

        results = torch.cat((source_img, target_img, swap_img), dim=0)
        save_image(results, os.path.join(self.args.log_dir, "image", "compare.png"))

        save_image(source_img, os.path.join(self.args.log_dir, "image", "source.png"))
        save_image(target_img, os.path.join(self.args.log_dir, "image", "target.png"))
        save_image(swap_img, os.path.join(self.args.log_dir, "image", "swap.png"))

    def _get_shifted_imgs(self, img: torch.tensor) -> torch.tensor:
        shifted_img = torch.roll(img.clone().detach(), shifts=(-1, 1), dims=(2, 3))
        rotated_img = torchvision.transforms.functional.rotate(shifted_img, 330)

        return rotated_img

    def _center_crop(self, img: np.array, dim: list[int]):
        width, height = img.shape[1], img.shape[0]

        crop_width = dim[0] if dim[0] < img.shape[1] else img.shape[1]
        crop_height = dim[1] if dim[1] < img.shape[0] else img.shape[0]
        mid_x, mid_y = int(width / 2), int(height / 2)
        cw2, ch2 = int(crop_width / 2), int(crop_height / 2)
        crop_img = img[mid_y - ch2 : mid_y + ch2, mid_x - cw2 : mid_x + cw2]
        return crop_img

    def _save_checkpoint(
        self,
        args,
        path: str,
        GAN_G_optim: torch.optim,
        GAN_G_loss: torch.tensor,
    ) -> None:
        torch.save(
            {
                "epoch": args.epoch,
                "GAN_G_state_dict": self.GAN_G.state_dict(),
                "GAN_G_optim": GAN_G_optim.state_dict(),
                "GAN_G_loss": GAN_G_loss,
            },
            path,
        )

    def pgd_src_single(self, loss_weights=[1, 1, 1]):
        self.logger.info(f"loss_weights: {loss_weights}")

        self.target.cuda().eval()
        l2_loss = nn.MSELoss().cuda()

        source_img = self._load_img(self.args.pgd_source)
        mimic_img = self._load_img(self.args.pgd_mimic)
        target_img = self._load_img(self.args.pgd_target)

        source_identity = self._get_img_identity(source_img)
        mimic_identity = self._get_img_identity(mimic_img)

        source_swap_img = self.target(None, target_img, source_identity, None, True)
        mimic_swap_img = self.target(None, target_img, mimic_identity, None, True)
        raw_results = torch.cat(
            (source_img, mimic_img, target_img, source_swap_img, mimic_swap_img), 0
        )

        x_img = source_img.clone().detach()
        x_backup = source_img.clone().detach()
        epsilon = (
            self.args.pgd_epsilon * (torch.max(source_img) - torch.min(source_img)) / 2
        )
        for iter in range(self.args.pgd_epochs):
            x_img.requires_grad = True

            x_identity = self._get_img_identity(x_img)
            x_swap_img = self.target(
                None, target_img.detach(), x_identity.detach(), None, True
            )

            pert_diff_loss = l2_loss(x_img, x_backup.detach())
            swap_diff_loss = l2_loss(x_swap_img, mimic_swap_img.detach())
            identity_diff_loss = l2_loss(x_identity, mimic_identity.detach())

            loss = (
                loss_weights[0] * pert_diff_loss
                + loss_weights[1] * swap_diff_loss
                + loss_weights[2] * identity_diff_loss
            )
            loss.backward(retain_graph=True)

            x_img = x_img.clone().detach() + epsilon * x_img.grad.sign()
            x_img = torch.clamp(
                x_img,
                min=x_backup - self.args.pgd_limit,
                max=x_backup + self.args.pgd_limit,
            )
            self.logger.info(
                f"[Iter {iter:4}]loss: {loss:.5f}({loss_weights[0] * pert_diff_loss.item():.5f}, {loss_weights[1] * swap_diff_loss.item():.5f}, {loss_weights[2] * identity_diff_loss.item():.5f})"
            )

        x_identity = self._get_img_identity(x_img).detach()
        x_swap_img = self.target(None, target_img, x_identity, None, True)
        protect_results = torch.cat((x_img, x_swap_img), 0)

        results = torch.cat((raw_results, protect_results), dim=0)
        save_image(
            results, os.path.join(self.args.log_dir, "image", "compare.png"), nrow=5
        )
        save_image(source_img, os.path.join(self.args.log_dir, "image", "source.png"))
        save_image(target_img, os.path.join(self.args.log_dir, "image", "target.png"))
        save_image(mimic_img, os.path.join(self.args.log_dir, "image", "mimic.png"))
        save_image(
            source_swap_img, os.path.join(self.args.log_dir, "image", "source_swap.png")
        )
        save_image(
            mimic_swap_img, os.path.join(self.args.log_dir, "image", "mimic_swap.png")
        )
        save_image(x_img, os.path.join(self.args.log_dir, "image", "pert.png"))
        save_image(
            x_swap_img, os.path.join(self.args.log_dir, "image", "pert_swap.png")
        )

    def pgd_src_multi(self, loss_weights=[1, 1, 1]) -> None:
        samples_list = os.listdir(os.path.join(self.args.data_dir, "samples"))
        samples_list.remove(self.args.pgd_mimic)
        source_imgs_path = [os.path.join("samples", img) for img in samples_list]

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
        src_prior = self._get_img_identity(src_imgs)
        tgt_prior = self._get_img_identity(tgt_imgs)

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

    def _get_shifted_imgs(self, img: torch.tensor) -> torch.tensor:
        shifted_img = torch.roll(img.clone().detach(), shifts=(-1, 1), dims=(2, 3))
        rotated_img = torchvision.transforms.functional.rotate(shifted_img, 330)

        return rotated_img

    def GAN_SRC(
        self,
        args,
        lr_g=5e-4,
        loss_ratio=[0.1, 1, 1],
        clipped=[0.01, 0.01],
    ):
        self.GAN_G.load_state_dict(self.target.netG.state_dict(), strict=False)
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
        for epoch in range(args.epoch):
            save_path = f"../log/{args.ID}/{args.project}_gan_src_{epoch}.png"

            self.GAN_G.to("cuda").train()

            src_imgs_path, dst_imgs_path = self._get_random_img_path(args.batch_size)

            src_imgs = self._load_src_imgs(src_imgs_path)
            src_prior = self._get_img_identity(src_imgs)
            dst_imgs = self._load_dst_imgs(dst_imgs_path)

            swapped_imgs = self.target(None, dst_imgs.detach(), src_prior, None, True)

            pert_src_imgs = self.GAN_G(src_imgs)
            pert_src_prior = self._get_img_identity(pert_src_imgs)
            pert_swapped_imgs = self.target(
                None, dst_imgs.detach(), pert_src_prior, None, True
            )

            self.GAN_G.zero_grad()
            pert_diff_loss = l2_loss(src_imgs, pert_src_imgs)
            swap_diff_loss = -torch.clamp(
                l2_loss(swapped_imgs, pert_swapped_imgs), 0.0, clipped[0]
            )
            prior_diff_loss = -torch.clamp(
                l2_loss(src_prior, pert_src_prior), 0.0, clipped[1]
            )

            G_loss = (
                loss_ratio[0] * pert_diff_loss
                + loss_ratio[1] * swap_diff_loss
                + loss_ratio[2] * prior_diff_loss
            )
            G_loss.backward()
            optimizer_G.step()
            scheduler.step()

            self.logger.info(
                f"[Epoch {epoch:4}]loss: {G_loss:.5f}({pert_diff_loss:.5f}, {swap_diff_loss:.5f}, {prior_diff_loss:.5f})"
            )

            if epoch % args.defense_save_interval == 0:
                with torch.no_grad():
                    self.GAN_G.eval()
                    self.target.eval()

                    src_imgs_path = [
                        f"{self.project_path}/crop_224/zjl.jpg",
                    ]
                    dst_imgs_path = [
                        f"{self.project_path}/crop_224/zrf.jpg",
                    ]

                    src_imgs = self._load_src_imgs(src_imgs_path)
                    dst_imgs = self._load_dst_imgs(dst_imgs_path)
                    src_prior = self._get_img_identity(src_imgs)

                    src_swapped_img = self.target(None, dst_imgs, src_prior, None, True)
                    raw_results = torch.cat(
                        (
                            src_imgs,
                            dst_imgs,
                            src_swapped_img,
                        ),
                        0,
                    )

                    x_imgs = self.GAN_G(src_imgs)
                    x_prior = self._get_img_identity(x_imgs)
                    x_swapped_img = self.target(None, dst_imgs, x_prior, None, True)
                    protect_results = torch.cat((x_imgs, x_swapped_img), 0)

                    self.logger.info(
                        f"save the result at log/{args.ID}/{args.project}_gan_src_{epoch}.png"
                    )

                    results = torch.cat((raw_results, protect_results), dim=0)
                    save_image(results, save_path, nrow=3)

            if G_loss.data < best_loss:
                best_loss = G_loss.data
                log_save_path = f"../log/{args.ID}/{args.project}.pth"
                checkpoint_save_path = f"../checkpoint/{args.project}.pth"
                self._save_checkpoint(args, log_save_path, optimizer_G, G_loss)
                self._save_checkpoint(args, checkpoint_save_path, optimizer_G, G_loss)

    def GAN_DST(
        self,
        args,
        lr_g=5e-4,
        loss_ratio=[0.1, 0.9, 0.002, 0.0],
        clipped=[0.05, 2.0],
    ):
        self.GAN_G.load_state_dict(self.target.netG.state_dict(), strict=False)
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
        for epoch in range(args.epoch):
            save_path = f"../log/{args.ID}/{args.project}_gan{epoch}_dst.png"

            self.GAN_G.to("cuda").train()

            src_imgs_path, dst_imgs_path = self._get_random_img_path(args.batch_size)

            src_imgs = self._load_src_imgs(src_imgs_path)
            dst_imgs = self._load_dst_imgs(dst_imgs_path)
            src_prior = self._get_img_identity(src_imgs)

            swapped_imgs = self.target(None, dst_imgs, src_prior.detach(), None, True)
            dst_latent_code = self.target.netG.encoder(dst_imgs)

            pert_dst_imgs = self.GAN_G(dst_imgs)
            pert_swapped_imgs = self.target(
                None, pert_dst_imgs, src_prior.detach(), None, True
            )
            pert_dst_latent_code = self.target.netG.encoder(pert_dst_imgs)

            shift_dst_imgs = self._get_shifted_imgs(dst_imgs)
            shift_dst_latent_code = self.target.netG.encoder(shift_dst_imgs)

            self.GAN_G.zero_grad()
            pert_diff_loss = l2_loss(dst_imgs, pert_dst_imgs)
            swap_diff_loss = -torch.clamp(
                l2_loss(swapped_imgs, pert_swapped_imgs), 0.0, clipped[0]
            )
            latent_code_diff_loss = -torch.clamp(
                l2_loss(dst_latent_code, pert_dst_latent_code), 0.0, clipped[1]
            )
            shift_latent_code_diff_loss = l2_loss(
                pert_dst_latent_code, shift_dst_latent_code
            )

            G_loss = (
                loss_ratio[0] * pert_diff_loss
                + loss_ratio[1] * swap_diff_loss
                + loss_ratio[2] * latent_code_diff_loss
                + loss_ratio[3] * shift_latent_code_diff_loss
            )
            G_loss.backward()
            optimizer_G.step()
            scheduler.step()

            self.logger.info(
                f"[Epoch {epoch:4}]loss: {G_loss:.5f}({pert_diff_loss:.5f}, {swap_diff_loss:.5f}, {latent_code_diff_loss:.5f}, {shift_latent_code_diff_loss:.5f})"
            )

            if epoch % args.defense_save_interval == 0:
                with torch.no_grad():
                    self.GAN_G.eval()
                    self.target.eval()

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
                    dst_imgs = self._load_dst_imgs(dst_imgs_path)
                    src_prior = self._get_img_identity(src_imgs)

                    src_swapped_img = self.target(None, dst_imgs, src_prior, None, True)
                    raw_results = torch.cat(
                        (
                            src_imgs,
                            dst_imgs,
                            src_swapped_img,
                        ),
                        0,
                    )

                    x_imgs = self.GAN_G(dst_imgs)
                    x_swapped_img = self.target(None, x_imgs, src_prior, None, True)
                    protect_results = torch.cat((x_imgs, x_swapped_img), 0)

                    self.logger.info(
                        f"save the result at log/{args.ID}/{args.project}_gan{epoch}_dst.png"
                    )

                    results = torch.cat((raw_results, protect_results), dim=0)
                    save_image(results, save_path, nrow=3)

            if G_loss.data < best_loss:
                best_loss = G_loss.data
                log_save_path = f"../log/{args.ID}/{args.project}.pth"
                checkpoint_save_path = f"../checkpoint/{args.project}.pth"
                self._save_checkpoint(args, log_save_path, optimizer_G, G_loss)
                self._save_checkpoint(args, checkpoint_save_path, optimizer_G, G_loss)

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
        test_src_imgs, test_dst_imgs = self._get_random_img_path(15)
        test_img_id = self._load_src_imgs(test_src_imgs)
        test_img_att = self._load_dst_imgs(test_dst_imgs)
        test_prior = self._get_img_identity(test_img_id)
        test_swap_img = self.target(None, test_img_att, test_prior, None, True)

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

            src_imgs, dst_imgs = self._get_random_img_path(args.batch_size)
            save_path = f"../log/{args.ID}/{args.project}_gan_{epoch}.png"

            img_att = self._load_dst_imgs(dst_imgs)
            img_id = self._load_src_imgs(src_imgs)
            latent_id = self._get_img_identity(img_id)
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

            if epoch % args.defense_save_interval == 0:
                with torch.no_grad():
                    self.GAN_G.eval()
                    self.target.eval()
                    test_protect_img = self.GAN_G(test_img_att)
                    test_pert_swap_img = self.target(
                        None, test_protect_img, test_prior, None, True
                    )
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

                    results = torch.cat(
                        (
                            test_img_att,
                            test_swap_img,
                            test_protect_img,
                            test_pert_swap_img,
                        ),
                        dim=0,
                    )
                    save_image(results, save_path, nrow=15)

            if G_loss.data < best_loss:
                best_loss = G_loss.data
                log_save_path = f"../log/{args.ID}/{args.project}.pth"
                checkpoint_save_path = f"../checkpoint/{args.project}.pth"
                self._save_checkpoint(args, log_save_path, optimizer_G, G_loss)
                self._save_checkpoint(args, checkpoint_save_path, optimizer_G, G_loss)
