import os
import random
import time
import math
import cv2
import shutil

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
        self.trainset_dir = join(args.data_dir, "train")
        self.testset_dir = join(args.data_dir, "test")

        self.gan_rgb_limits = [0.075, 0.03, 0.075]
        self.gan_src_loss_limits = [0.01, 0.01]
        self.gan_tgt_loss_limits = [0.05, 7.5]
        self.gan_src_loss_weights = [60, 10, 0.1]  # pert, swap diff, identity diff
        self.gan_tgt_loss_weights = [
            30,
            10,
            0.1,
            0.025,
        ]  # pert, swap diff, latent diff, rotate latent diff

        self.target = create_model(TestOptions().parse())
        self.GAN_G = Generator(input_nc=3, output_nc=3, epsilon=self.gan_rgb_limits)

        self.utility = Utility()
        self.efficiency = Efficiency(None)

    def split_dataset(self) -> None:
        all_people = os.listdir(self.dataset_dir)
        testset_people = random.sample(all_people, self.args.testset_people_count)

        os.makedirs(self.trainset_dir, exist_ok=True)
        os.makedirs(self.testset_dir, exist_ok=True)

        for i, people in enumerate(all_people, start=1):
            self.logger.info(f"{i:4}/{len(all_people):4}|Copy folder {people}")
            shutil.copytree(
                join(self.dataset_dir, people),
                (
                    join(self.testset_dir, people)
                    if people in testset_people
                    else join(self.trainset_dir, people)
                ),
            )

    def _load_imgs(self, imgs_path: list[str]) -> torch.tensor:
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
        elif len(imgs) == 13:
            (
                source,
                target,
                swap,
                pert,
                pert_swap,
                noise_pert,
                noise_pert_swap,
                blur_pert,
                blur_pert_swap,
                compress_pert,
                compress_pert_swap,
                rotate_pert,
                rotate_pert_swap,
            ) = imgs[:]

            results = torch.cat(
                (
                    source,
                    target,
                    swap,
                    pert,
                    pert_swap,
                    noise_pert,
                    noise_pert_swap,
                    blur_pert,
                    blur_pert_swap,
                    compress_pert,
                    compress_pert_swap,
                    rotate_pert,
                    rotate_pert_swap,
                ),
                dim=0,
            )
            save_image(results, join(log_dir, "image", "compare.png"), nrow=3)

            for i, img in enumerate(source, 1):
                save_image(img, join(log_dir, "image", f"source_{i}.png"))
            for i, img in enumerate(target, 1):
                save_image(img, join(log_dir, "image", f"target_{i}.png"))
            for i, img in enumerate(swap, 1):
                save_image(img, join(log_dir, "image", f"swap_{i}.png"))
            for i, img in enumerate(pert, 1):
                save_image(img, join(log_dir, "image", f"pert_{i}.png"))
            for i, img in enumerate(pert_swap, 1):
                save_image(img, join(log_dir, "image", f"pert_swap_{i}.png"))
            for i, img in enumerate(noise_pert, 1):
                save_image(img, join(log_dir, "image", f"noise_pert_{i}.png"))
            for i, img in enumerate(noise_pert_swap, 1):
                save_image(img, join(log_dir, "image", f"noise_pert_swap_{i}.png"))
            for i, img in enumerate(blur_pert, 1):
                save_image(img, join(log_dir, "image", f"blur_pert_{i}.png"))
            for i, img in enumerate(blur_pert_swap, 1):
                save_image(img, join(log_dir, "image", f"blur_pert_swap_{i}.png"))
            for i, img in enumerate(compress_pert, 1):
                save_image(img, join(log_dir, "image", f"compress_pert_{i}.png"))
            for i, img in enumerate(compress_pert_swap, 1):
                save_image(img, join(log_dir, "image", f"compress_pert_swap_{i}.png"))
            for i, img in enumerate(rotate_pert, 1):
                save_image(img, join(log_dir, "image", f"rotate_pert_{i}.png"))
            for i, img in enumerate(rotate_pert_swap, 1):
                save_image(img, join(log_dir, "image", f"rotate_pert_swap_{i}.png"))

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

    def _get_all_imgs_path(self, train_set: bool = True) -> list[str]:
        set_to_load = self.trainset_dir if train_set else self.testset_dir
        all_people = os.listdir(set_to_load)
        all_imgs_path = []
        for people in all_people:
            people_dir = join(set_to_load, people)
            all_imgs_name = os.listdir(people_dir)
            all_imgs_path.extend(
                [join(set_to_load, people, name) for name in all_imgs_name]
            )

        self.logger.info(
            f"Collect {len(all_imgs_path)} images for GAN {'training' if train_set else 'test'}"
        )
        return all_imgs_path

    def GAN_SRC(self):
        self.logger.info(
            f"rgb_limits: {self.gan_rgb_limits}, loss_limits: {self.gan_src_loss_limits}, loss_weights: {self.gan_src_loss_weights}"
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
        train_imgs_path = self._get_all_imgs_path(train_set=True)
        test_imgs_path = self._get_all_imgs_path(train_set=False)
        for epoch in range(self.args.gan_epochs):
            self.GAN_G.cuda().train()
            src_imgs_path = random.sample(train_imgs_path, self.args.gan_batch_size)
            tgt_imgs_path = random.sample(train_imgs_path, self.args.gan_batch_size)

            src_imgs = self._load_imgs(src_imgs_path)
            src_identity = self._get_imgs_identity(src_imgs)
            tgt_imgs = self._load_imgs(tgt_imgs_path)
            tgt_swap_imgs = self.target(None, tgt_imgs, src_identity, None, True)

            pert_src_imgs = self.GAN_G(src_imgs)
            pert_src_identity = self._get_imgs_identity(pert_src_imgs)
            pert_swap_imgs = self.target(None, tgt_imgs, pert_src_identity, None, True)

            self.GAN_G.zero_grad()

            pert_diff_loss = l2_loss(flatten(pert_src_imgs), flatten(src_imgs))
            swap_diff_loss = -torch.clamp(
                l2_loss(flatten(pert_swap_imgs), flatten(tgt_swap_imgs)),
                0.0,
                self.gan_src_loss_limits[0],
            )
            identity_diff_loss = -torch.clamp(
                l2_loss(flatten(pert_src_identity), flatten(src_identity)),
                0.0,
                self.gan_src_loss_limits[1],
            )

            G_loss = (
                self.gan_src_loss_weights[0] * pert_diff_loss
                + self.gan_src_loss_weights[1] * swap_diff_loss
                + self.gan_src_loss_weights[2] * identity_diff_loss
            )
            G_loss.backward()
            optimizer_G.step()
            scheduler.step()

            self.logger.info(
                f"[Epoch {epoch:6}]loss: {G_loss:8.5f}({self.gan_src_loss_weights[0] * pert_diff_loss.item():.5f}, {self.gan_src_loss_weights[1] * swap_diff_loss.item():.5f}, {self.gan_src_loss_weights[2] * identity_diff_loss.item():.5f})({swap_diff_loss.item():.5f}, {identity_diff_loss.item():.5f})"
            )

            if epoch % self.args.gan_generator_interval == 0:
                with torch.no_grad():
                    self.GAN_G.eval()
                    self.target.eval()

                    src_imgs_path = random.sample(test_imgs_path, 7)
                    src_imgs_path.extend(
                        [
                            join(self.samples_dir, "zjl.jpg"),
                            join(self.samples_dir, "6.jpg"),
                            join(self.samples_dir, "jl.jpg"),
                        ]
                    )
                    tgt_imgs_path = random.sample(test_imgs_path, 7)
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
                    raw_results = torch.cat((src_imgs, tgt_imgs, src_swap_img), 0)

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
            f"rgb_limits: {self.gan_rgb_limits}, loss_limits: {self.gan_tgt_loss_limits}, loss_weights: {self.gan_tgt_loss_weights}"
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
        train_imgs_path = self._get_all_imgs_path(train_set=True)
        test_imgs_path = self._get_all_imgs_path(train_set=False)
        for epoch in range(self.args.gan_epochs):
            self.GAN_G.cuda().train()
            src_imgs_path = random.sample(train_imgs_path, self.args.gan_batch_size)
            tgt_imgs_path = random.sample(train_imgs_path, self.args.gan_batch_size)

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
                self.gan_tgt_loss_limits[0],
            )
            latent_diff_loss = -torch.clamp(
                l2_loss(flatten(pert_latent_code), flatten(tgt_latent_code)),
                0.0,
                self.gan_tgt_loss_limits[1],
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

                    src_imgs_path = random.sample(test_imgs_path, 7)
                    src_imgs_path.extend(
                        [
                            join(self.samples_dir, "zrf.jpg"),
                            join(self.samples_dir, "zrf.jpg"),
                            join(self.samples_dir, "zrf.jpg"),
                        ]
                    )
                    tgt_imgs_path = random.sample(test_imgs_path, 7)
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

    def _get_split_test_imgs_path(self) -> tuple[list[str], list[str]]:
        all_people = os.listdir(self.testset_dir)
        random.shuffle(all_people)

        source_people = all_people[: int(len(all_people) / 2)]
        target_people = all_people[int(len(all_people) / 2) :]

        source_imgs_path = []
        for people in source_people:
            people_dir = join(self.testset_dir, people)
            people_imgs_name = os.listdir(people_dir)
            source_imgs_path.extend(
                [join(self.testset_dir, people, name) for name in people_imgs_name]
            )

        target_imgs_path = []
        for people in target_people:
            people_dir = join(self.testset_dir, people)
            people_imgs_name = os.listdir(people_dir)
            target_imgs_path.extend(
                [join(self.testset_dir, people, name) for name in people_imgs_name]
            )

        return source_imgs_path, target_imgs_path

    def GAN_SRC_test(self):
        from utils import calculate_score

        model_path = join("checkpoints", self.args.gan_test_models)
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

        source_imgs_path, target_imgs_path = self._get_split_test_imgs_path()
        utilities, clean_efficiencies, pert_efficiencies = [], [], []
        scores = []
        total_batch = (
            min(len(source_imgs_path), len(target_imgs_path))
            // self.args.gan_batch_size
        )
        for i in range(total_batch):
            iter_source_path = source_imgs_path[
                i * self.args.gan_batch_size : (i + 1) * self.args.gan_batch_size
            ]
            iter_target_path = target_imgs_path[
                i * self.args.gan_batch_size : (i + 1) * self.args.gan_batch_size
            ]

            source_imgs = self._load_imgs(iter_source_path)
            target_imgs = self._load_imgs(iter_target_path)
            source_identity = self._get_imgs_identity(source_imgs)
            swap_imgs = self.target(None, target_imgs, source_identity, None, True)

            pert_source_imgs = self.GAN_G(source_imgs)
            pert_source_identity = self._get_imgs_identity(pert_source_imgs)
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
            f"Average of {self.args.gan_batch_size * total_batch} pictures: utility: {sum(utilities)/len(utilities):.3f}, efficiency: {sum(clean_efficiencies)/len(clean_efficiencies):.3f}, {sum(pert_efficiencies)/len(pert_efficiencies):.3f}, score: {sum(scores)/len(scores):.3f}"
        )

    def GAN_TGT_test(self):
        from utils import calculate_score

        model_path = join("checkpoints", self.args.gan_test_models)
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

        source_imgs_path, target_imgs_path = self._get_split_test_imgs_path()
        utilities, clean_efficiencies, pert_efficiencies = [], [], []
        scores = []
        total_batch = (
            min(len(source_imgs_path), len(target_imgs_path))
            // self.args.gan_batch_size
        )
        for i in range(total_batch):
            iter_source_path = source_imgs_path[
                i * self.args.gan_batch_size : (i + 1) * self.args.gan_batch_size
            ]
            iter_target_path = target_imgs_path[
                i * self.args.gan_batch_size : (i + 1) * self.args.gan_batch_size
            ]

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
                f"Iter {i:4}/{total_batch:4}, utility: {utility:.3f}, efficiency: {source_clean_swap:.3f}, {source_pert_swap:.3f}, score: {score:.3f}"
            )

        self.logger.info(
            f"Average of {total_batch:4} batch and {self.args.gan_batch_size * total_batch:6} pictures: utility: {sum(utilities)/len(utilities):.3f}, efficiency: {sum(clean_efficiencies)/len(clean_efficiencies):.3f}, {sum(pert_efficiencies)/len(pert_efficiencies):.3f}, score: {sum(scores)/len(scores):.3f}"
        )

    def _gauss_noise(
        self, pert: torch.tensor, gauss_mean: float, gauss_std: float
    ) -> torch.tensor:
        gauss_noise = gauss_mean + gauss_std * torch.randn(pert.shape).cuda()
        noise_pert = pert + gauss_noise

        return noise_pert

    def _gauss_kernel(self, size: int, sigma: float):
        coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2.0
        grid = coords.repeat(size).view(size, size)
        kernel = torch.exp(-0.5 * (grid**2 + grid.t() ** 2) / sigma**2)
        kernel = kernel / kernel.sum()

        return kernel

    def _gauss_blur(self, pert: torch.tensor, size: int, sigma: float) -> torch.tensor:
        kernel = self._gauss_kernel(size, sigma).cuda()
        kernel = kernel.view(1, 1, size, size)
        kernel = kernel.repeat(pert.shape[1], 1, 1, 1)
        blurred_pert = F.conv2d(pert, kernel, padding=size // 2, groups=pert.shape[1])

        return blurred_pert.squeeze(0)

    def _jpeg_compress(self, pert: torch.tensor, ratio: int) -> torch.tensor:
        pert_np = pert.detach().cpu().numpy().transpose(0, 2, 3, 1)
        pert_np = np.clip(pert_np * 255.0, 0, 255).astype("uint8")
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(ratio)]
        for i in range(pert_np.shape[0]):
            _, encimg = cv2.imencode(".jpg", pert_np[i], encode_params)
            pert_np[i] = cv2.imdecode(encimg, 1)

        compress_pert = pert_np.transpose((0, 3, 1, 2))
        compress_pert = torch.from_numpy(compress_pert / 255.0).float().cuda()

        return compress_pert

    def _rotate(self, pert: torch.tensor, angle: float) -> None:
        import torchvision.transforms.functional as F

        rotated_tensor = torch.stack([F.rotate(tensor, angle) for tensor in pert])

        return rotated_tensor

    def GAN_SRC_robust(self) -> None:
        gauss_mean, gauss_std = 0, 0.1
        gauss_size, gauss_sigma = 5, 3.0
        jpeg_ratio = 70

        model_path = join("checkpoints", self.args.gan_test_models)
        self.GAN_G.load_state_dict(torch.load(model_path)["GAN_G_state_dict"])

        self.target.cuda().eval()
        self.GAN_G.cuda().eval()

        source_path = [
            join(self.samples_dir, "zjl.jpg"),
            join(self.samples_dir, "6.jpg"),
            join(self.samples_dir, "hzxc.jpg"),
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

        pert_imgs = self.GAN_G(source_imgs)
        pert_identity = self._get_imgs_identity(pert_imgs)
        pert_swap_imgs = self.target(None, target_imgs, pert_identity, None, True)

        noise_imgs = self._gauss_noise(pert_imgs, gauss_mean, gauss_std)
        noise_identity = self._get_imgs_identity(noise_imgs)
        noise_swap_imgs = self.target(None, target_imgs, noise_identity, None, True)

        blur_imgs = self._gauss_blur(pert_imgs, gauss_size, gauss_sigma)
        blur_identity = self._get_imgs_identity(blur_imgs)
        blur_swap_imgs = self.target(None, target_imgs, blur_identity, None, True)

        compress_imgs = self._jpeg_compress(pert_imgs, jpeg_ratio)
        compress_identity = self._get_imgs_identity(compress_imgs)
        compress_swap_imgs = self.target(
            None, target_imgs, compress_identity, None, True
        )

        rotate_imgs = self._rotate(pert_imgs, random.random() * 360)
        rotate_identity = self._get_imgs_identity(rotate_imgs)
        rotate_swap_imgs = self.target(None, target_imgs, rotate_identity, None, True)
        self._save_imgs(
            [
                source_imgs,
                target_imgs,
                swap_imgs,
                pert_imgs,
                pert_swap_imgs,
                noise_imgs,
                noise_swap_imgs,
                blur_imgs,
                blur_swap_imgs,
                compress_imgs,
                compress_swap_imgs,
                rotate_imgs,
                rotate_swap_imgs,
            ]
        )

        source_imgs_path, target_imgs_path = self._get_split_test_imgs_path()
        efficiencies = {
            "clean": [],
            "pert": [],
            "noise": [],
            "blur": [],
            "compress": [],
            "rotate": [],
        }
        total_batch = (
            min(len(source_imgs_path), len(target_imgs_path))
            // self.args.gan_batch_size
        )
        for i in range(total_batch):
            iter_source_path = source_imgs_path[
                i * self.args.gan_batch_size : (i + 1) * self.args.gan_batch_size
            ]
            iter_target_path = target_imgs_path[
                i * self.args.gan_batch_size : (i + 1) * self.args.gan_batch_size
            ]

            source_imgs = self._load_imgs(iter_source_path)
            target_imgs = self._load_imgs(iter_target_path)
            source_identity = self._get_imgs_identity(source_imgs)
            swap_imgs = self.target(None, target_imgs, source_identity, None, True)

            pert_imgs = self.GAN_G(target_imgs)
            pert_identity = self._get_imgs_identity(pert_imgs)
            pert_swap_imgs = self.target(None, target_imgs, pert_identity, None, True)

            noise_imgs = self._gauss_noise(pert_imgs, gauss_mean, gauss_std)
            noise_identity = self._get_imgs_identity(noise_imgs)
            noise_swap_imgs = self.target(None, target_imgs, noise_identity, None, True)

            blur_imgs = self._gauss_blur(pert_imgs, gauss_size, gauss_sigma)
            blur_identity = self._get_imgs_identity(blur_imgs)
            blur_swap_imgs = self.target(None, target_imgs, blur_identity, None, True)

            compress_imgs = self._jpeg_compress(pert_imgs, jpeg_ratio)
            compress_identity = self._get_imgs_identity(compress_imgs)
            compress_swap_imgs = self.target(
                None, target_imgs, compress_identity, None, True
            )

            rotate_imgs = self._rotate(pert_imgs, random.random() * 360)
            rotate_identity = self._get_imgs_identity(rotate_imgs)
            rotate_swap_imgs = self.target(
                None, target_imgs, rotate_identity, None, True
            )

            clean_swap_effi, pert_swap_effi = self._calculate_efficiency(
                source_imgs, swap_imgs, pert_swap_imgs
            )
            efficiencies["clean"].append(clean_swap_effi)
            efficiencies["pert"].append(pert_swap_effi)

            _, noise_swap_effi = self._calculate_efficiency(
                source_imgs, swap_imgs, noise_swap_imgs
            )
            efficiencies["noise"].append(noise_swap_effi)

            _, blur_swap_effi = self._calculate_efficiency(
                source_imgs, swap_imgs, blur_swap_imgs
            )
            efficiencies["blur"].append(blur_swap_effi)

            _, compress_swap_effi = self._calculate_efficiency(
                source_imgs, swap_imgs, compress_swap_imgs
            )
            efficiencies["compress"].append(compress_swap_effi)

            _, rotate_swap_effi = self._calculate_efficiency(
                source_imgs, swap_imgs, rotate_swap_imgs
            )
            efficiencies["rotate"].append(rotate_swap_effi)

            del (
                swap_imgs,
                pert_swap_imgs,
                noise_imgs,
                noise_swap_imgs,
                blur_imgs,
                blur_swap_imgs,
                compress_imgs,
                compress_swap_imgs,
                rotate_imgs,
                rotate_swap_imgs,
            )

            utility = self._calculate_utility(target_imgs, pert_imgs)

            self.logger.info(
                f"Iter {i:4}/{total_batch:4}, utility: {utility:.3f}, efficiency: {clean_swap_effi:.3f}, {pert_swap_effi:.3f}, {noise_swap_effi:.3f}, {blur_swap_effi:.3f}, {compress_swap_effi:.3f}, {rotate_swap_effi:.3f}"
            )

            torch.cuda.empty_cache()

        self.logger.info(
            f"Average of {total_batch:4} batch and {self.args.gan_batch_size * total_batch:6} pictures: efficiency: {sum(efficiencies['clean'])/len(efficiencies['clean']):.3f}, {sum(efficiencies['pert'])/len(efficiencies['pert']):.3f}, {sum(efficiencies['noise'])/len(efficiencies['noise']):.3f}, {sum(efficiencies['blur'])/len(efficiencies['blur']):.3f}, {sum(efficiencies['compress'])/len(efficiencies['compress']):.3f}, {sum(efficiencies['rotate'])/len(efficiencies['rotate']):.3f}"
        )

    def GAN_TGT_robust(self) -> None:
        gauss_mean, gauss_std = 0, 0.1
        gauss_size, gauss_sigma = 5, 3.0
        jpeg_ratio = 70

        model_path = join("checkpoints", self.args.gan_test_models)
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
            join(self.samples_dir, "hzxc.jpg"),
        ]

        source_imgs = self._load_imgs(source_path)
        target_imgs = self._load_imgs(target_path)
        source_identity = self._get_imgs_identity(source_imgs)
        swap_imgs = self.target(None, target_imgs, source_identity, None, True)

        pert_imgs = self.GAN_G(target_imgs)
        pert_swap_imgs = self.target(None, pert_imgs, source_identity, None, True)

        noise_imgs = self._gauss_noise(pert_imgs, gauss_mean, gauss_std)
        noise_swap_imgs = self.target(None, noise_imgs, source_identity, None, True)

        blur_imgs = self._gauss_blur(pert_imgs, gauss_size, gauss_sigma)
        blur_swap_imgs = self.target(None, blur_imgs, source_identity, None, True)

        compress_imgs = self._jpeg_compress(pert_imgs, jpeg_ratio)
        compress_swap_imgs = self.target(
            None, compress_imgs, source_identity, None, True
        )

        rotate_imgs = self._rotate(pert_imgs, random.random() * 360)
        rotate_swap_imgs = self.target(None, rotate_imgs, source_identity, None, True)
        self._save_imgs(
            [
                source_imgs,
                target_imgs,
                swap_imgs,
                pert_imgs,
                pert_swap_imgs,
                noise_imgs,
                noise_swap_imgs,
                blur_imgs,
                blur_swap_imgs,
                compress_imgs,
                compress_swap_imgs,
                rotate_imgs,
                rotate_swap_imgs,
            ]
        )

        source_imgs_path, target_imgs_path = self._get_split_test_imgs_path()
        efficiencies = {
            "clean": [],
            "pert": [],
            "noise": [],
            "blur": [],
            "compress": [],
            "rotate": [],
        }
        total_batch = (
            min(len(source_imgs_path), len(target_imgs_path))
            // self.args.gan_batch_size
        )
        for i in range(total_batch):
            iter_source_path = source_imgs_path[
                i * self.args.gan_batch_size : (i + 1) * self.args.gan_batch_size
            ]
            iter_target_path = target_imgs_path[
                i * self.args.gan_batch_size : (i + 1) * self.args.gan_batch_size
            ]

            source_imgs = self._load_imgs(iter_source_path)
            target_imgs = self._load_imgs(iter_target_path)
            source_identity = self._get_imgs_identity(source_imgs)
            swap_imgs = self.target(None, target_imgs, source_identity, None, True)

            pert_imgs = self.GAN_G(target_imgs)
            pert_swap_imgs = self.target(None, pert_imgs, source_identity, None, True)

            noise_imgs = self._gauss_noise(pert_imgs, gauss_mean, gauss_std)
            noise_swap_imgs = self.target(None, noise_imgs, source_identity, None, True)

            blur_imgs = self._gauss_blur(pert_imgs, gauss_size, gauss_sigma)
            blur_swap_imgs = self.target(None, blur_imgs, source_identity, None, True)

            compress_imgs = self._jpeg_compress(pert_imgs, jpeg_ratio)
            compress_swap_imgs = self.target(
                None, compress_imgs, source_identity, None, True
            )

            rotate_imgs = self._rotate(pert_imgs, random.random() * 360)
            rotate_swap_imgs = self.target(
                None, rotate_imgs, source_identity, None, True
            )

            clean_swap_effi, pert_swap_effi = self._calculate_efficiency(
                source_imgs, swap_imgs, pert_swap_imgs
            )
            efficiencies["clean"].append(clean_swap_effi)
            efficiencies["pert"].append(pert_swap_effi)

            _, noise_swap_effi = self._calculate_efficiency(
                source_imgs, swap_imgs, noise_swap_imgs
            )
            efficiencies["noise"].append(noise_swap_effi)

            _, blur_swap_effi = self._calculate_efficiency(
                source_imgs, swap_imgs, blur_swap_imgs
            )
            efficiencies["blur"].append(blur_swap_effi)

            _, compress_swap_effi = self._calculate_efficiency(
                source_imgs, swap_imgs, compress_swap_imgs
            )
            efficiencies["compress"].append(compress_swap_effi)

            _, rotate_swap_effi = self._calculate_efficiency(
                source_imgs, swap_imgs, rotate_swap_imgs
            )
            efficiencies["rotate"].append(rotate_swap_effi)

            del (
                swap_imgs,
                pert_swap_imgs,
                noise_imgs,
                noise_swap_imgs,
                blur_imgs,
                blur_swap_imgs,
                compress_imgs,
                compress_swap_imgs,
                rotate_imgs,
                rotate_swap_imgs,
            )

            utility = self._calculate_utility(target_imgs, pert_imgs)

            self.logger.info(
                f"Iter {i:4}/{total_batch:4}, utility: {utility:.3f}, efficiency: {clean_swap_effi:.3f}, {pert_swap_effi:.3f}, {noise_swap_effi:.3f}, {blur_swap_effi:.3f}, {compress_swap_effi:.3f}, {rotate_swap_effi:.3f}"
            )

            torch.cuda.empty_cache()

        self.logger.info(
            f"Average of {total_batch:4} batch and {self.args.gan_batch_size * total_batch:6} pictures: efficiency: {sum(efficiencies['clean'])/len(efficiencies['clean']):.3f}, {sum(efficiencies['pert'])/len(efficiencies['pert']):.3f}, {sum(efficiencies['noise'])/len(efficiencies['noise']):.3f}, {sum(efficiencies['blur'])/len(efficiencies['blur']):.3f}, {sum(efficiencies['compress'])/len(efficiencies['compress']):.3f}, {sum(efficiencies['rotate'])/len(efficiencies['rotate']):.3f}"
        )
