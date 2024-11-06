import os
import random

import cv2

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

from evaluate import Utility, Effectiveness


class SimSwapDefense(nn.Module):
    def __init__(self, args, logger):
        super(SimSwapDefense, self).__init__()
        self.args = args
        self.logger = logger

        self.samples_dir = join(args.data_dir, "samples")
        self.dataset_dir = join(args.data_dir, "vggface2_crop_224")
        self.trainset_dir = join(args.data_dir, "train")
        self.testset_dir = join(args.data_dir, "test")
        self.anchorset_dir = join(args.data_dir, "anchor")

        self.gan_rgb_limits = [0.075, 0.03, 0.075]
        self.gan_src_loss_limits = [0.01, 0.01]
        self.gan_tgt_loss_limits = [0.05, 7.5]
        self.gan_src_loss_weights = [
            60,
            10,
            0.1,
            15,
        ]  # pert, swap diff, identity diff, identity mimic
        self.gan_tgt_loss_weights = [
            30,
            10,
            0.1,
            0.025,
        ]  # pert, swap diff, latent diff, rotate latent diff

        self.target = create_model(TestOptions().parse())
        self.GAN_G = Generator(input_nc=3, output_nc=3, epsilon=self.gan_rgb_limits)

        self.utility = Utility()
        self.effectiveness = Effectiveness()

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

    def __save_pgd_samples(self, imgs: list[torch.tensor]) -> None:
        img_names = [
            "source",
            "target",
            "swap",
            "mimic",
            "mimic_swap",
            "pert",
            "pert_swap",
        ]

        for i, name in enumerate(img_names):
            for j in range(imgs[i].shape[0]):
                save_image(
                    imgs[i][j], join(self.args.log_dir, "image", f"{name}_{j}.png")
                )

        results = torch.cat(imgs, dim=0)
        save_image(
            results,
            join(self.args.log_dir, "image", f"summary.png"),
            nrow=imgs[0].shape[0],
        )
        del results

    def pgd_source_sample(self, loss_weights=[1, 1]):
        self.logger.info(f"loss_weights: {loss_weights}")

        self.target.cuda().eval()
        l2_loss = nn.MSELoss().cuda()

        source_path = [
            join(self.samples_dir, "zjl.jpg"),
            # join(self.samples_dir, "6.jpg"),
            # join(self.samples_dir, "hzxc.jpg"),
        ]
        target_path = [
            # join(self.samples_dir, "zrf.jpg"),
            # join(self.samples_dir, "zrf.jpg"),
            join(self.samples_dir, "zrf.jpg"),
        ]

        source_imgs = self._load_imgs(source_path)
        target_imgs = self._load_imgs(target_path)
        source_identity = self._get_imgs_identity(source_imgs)

        mimic_img = self._load_imgs([join(self.args.data_dir, self.args.pgd_mimic)])
        mimic_img_expand = mimic_img.repeat(len(source_path), 1, 1, 1)
        mimic_identity_expand = self._get_imgs_identity(mimic_img_expand)

        x_imgs = source_imgs.clone().detach()
        x_backup = source_imgs.clone().detach()
        epsilon = (
            self.args.pgd_epsilon
            * (torch.max(source_imgs) - torch.min(source_imgs))
            / 2
        )
        for iter in range(self.args.epochs):
            x_imgs.requires_grad = True

            x_identity = self._get_imgs_identity(x_imgs)
            pert_diff_loss = l2_loss(x_imgs, x_backup.detach())
            identity_diff_loss = l2_loss(x_identity, mimic_identity_expand.detach())
            loss = (
                loss_weights[0] * pert_diff_loss + loss_weights[1] * identity_diff_loss
            )

            loss.backward(retain_graph=True)

            x_imgs = (
                x_imgs.clone().detach() - epsilon * x_imgs.grad.sign().clone().detach()
            )
            x_imgs = torch.clamp(
                x_imgs,
                min=x_backup - self.args.pgd_limit,
                max=x_backup + self.args.pgd_limit,
            )
            self.logger.info(
                f"[Iter {iter:4}]loss: {loss:.5f}({loss_weights[0] * pert_diff_loss.item():.5f}, {loss_weights[1] * identity_diff_loss.item():.5f})"
            )

        source_swap_imgs = self.target(None, target_imgs, source_identity, None, True)
        mimic_swap_imgs = self.target(
            None, target_imgs, mimic_identity_expand, None, True
        )

        x_identity = self._get_imgs_identity(x_imgs).detach()
        x_swap_imgs = self.target(None, target_imgs, x_identity, None, True)

        self.__save_pgd_samples(
            [
                source_imgs,
                target_imgs,
                source_swap_imgs,
                mimic_img_expand,
                mimic_swap_imgs,
                x_imgs,
                x_swap_imgs,
            ]
        )

        pert_mse, pert_psnr, pert_ssim = self._calculate_utility(source_imgs, x_imgs)
        pert_swap_mse, pert_swap_psnr, pert_swap_ssim = self._calculate_utility(
            source_swap_imgs, x_swap_imgs
        )
        (
            pert_effectiveness,
            swap_effectiveness,
            pert_swap_effectiveness,
            anchor_effectiveness,
        ) = self._calculate_effectiveness(
            source_imgs,
            None,
            x_imgs,
            source_swap_imgs,
            x_swap_imgs,
            mimic_img_expand,
        )

        (
            source_to_swap_dist,
            target_to_swap_dist,
            anchor_to_swap_dist,
        ) = self._calculate_distance(
            source_imgs, target_imgs, mimic_img_expand, source_swap_imgs
        )
        (
            source_to_pert_swap_dist,
            target_to_pert_swap_dist,
            anchor_to_pert_swap_dist,
        ) = self._calculate_distance(
            source_imgs, target_imgs, mimic_img_expand, x_swap_imgs
        )

        self.logger.info(
            f"pert(mse, psnr, ssim): ({pert_mse:.5f}, {pert_psnr:.5f}, {pert_ssim:.5f}). pert swap(mse, psnr, ssim): ({pert_swap_mse:.5f}, {pert_swap_psnr:.5f}, {pert_swap_ssim:.5f}). effectiveness(pert, swap, pert swap, anchor): ({pert_effectiveness:.5f}, {swap_effectiveness:.5f}, {pert_swap_effectiveness:.5f}, {anchor_effectiveness:.5f}), distance(swap to source, target, anchor): ({source_to_swap_dist:.5f}, {target_to_swap_dist:.5f}, {anchor_to_swap_dist:.5f}), distance(pert swap to source, target, anchor): ({source_to_pert_swap_dist:.5f}, {target_to_pert_swap_dist:.5f}, {anchor_to_pert_swap_dist:.5f})"
        )

    def pgd_target_sample(self, loss_weights=[5000, 1]):
        self.logger.info(f"loss_weights: {loss_weights}")

        self.target.cuda().eval()
        l2_loss = nn.MSELoss().cuda()

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

        mimic_img = self._load_imgs([join(self.args.data_dir, self.args.pgd_mimic)])
        mimic_img_expand = mimic_img.repeat(len(source_path), 1, 1, 1)

        x_imgs = target_imgs.clone().detach()
        epsilon = (
            self.args.pgd_epsilon
            * (torch.max(target_imgs) - torch.min(target_imgs))
            / 2
        )
        for epoch in range(self.args.epochs):
            x_imgs.requires_grad = True

            x_latent_code = self.target.netG.encoder(x_imgs)
            mimic_latent_code = self.target.netG.encoder(mimic_img_expand)
            pert_diff_loss = l2_loss(x_imgs, target_imgs.detach())
            latent_code_diff_loss = l2_loss(x_latent_code, mimic_latent_code.detach())
            loss = (
                loss_weights[0] * pert_diff_loss
                - loss_weights[1] * latent_code_diff_loss
            )

            loss.backward(retain_graph=True)

            x_imgs = (
                x_imgs.clone().detach() - epsilon * x_imgs.grad.sign().clone().detach()
            )
            x_imgs = torch.clamp(
                x_imgs,
                min=target_imgs - self.args.pgd_limit,
                max=target_imgs + self.args.pgd_limit,
            )
            self.logger.info(
                f"[Epoch {epoch:4}]loss: {loss:.5f}({loss_weights[0] * pert_diff_loss.item():.5f}, {loss_weights[1] * latent_code_diff_loss.item():.5f})"
            )

        target_swap_imgs = self.target(None, target_imgs, source_identity, None, True)
        mimic_swap_imgs = self.target(
            None, mimic_img_expand, source_identity, None, True
        )

        x_swap_imgs = self.target(None, x_imgs, source_identity, None, True)

        self.__save_pgd_samples(
            [
                source_imgs,
                target_imgs,
                target_swap_imgs,
                mimic_img_expand,
                mimic_swap_imgs,
                x_imgs,
                x_swap_imgs,
            ]
        )

        pert_mse, pert_psnr, pert_ssim = self._calculate_utility(target_imgs, x_imgs)
        pert_swap_mse, pert_swap_psnr, pert_swap_ssim = self._calculate_utility(
            target_swap_imgs, x_swap_imgs
        )
        (pert_effectiveness, swap_effectiveness, pert_swap_effectiveness) = (
            self._calculate_effectiveness(
                source_imgs,
                target_imgs,
                x_imgs,
                target_swap_imgs,
                x_swap_imgs,
                None,
            )
        )

        self.logger.info(
            f"pert(mse, psnr, ssim): ({pert_mse:.5f}, {pert_psnr:.5f}, {pert_ssim:.5f}). pert swap(mse, psnr, ssim): ({pert_swap_mse:.5f}, {pert_swap_psnr:.5f}, {pert_swap_ssim:.5f}). effectiveness(pert, swap, pert swap, anchor): ({pert_effectiveness:.5f}, {swap_effectiveness:.5f}, {pert_swap_effectiveness:.5f})"
        )

    def __get_face_mask(
        self, imgs: torch.tensor, face_ratio: int = 0.2
    ) -> torch.tensor:
        from facenet_pytorch import MTCNN
        from PIL import Image

        mtcnn = MTCNN(keep_all=True)
        mask = torch.ones_like(imgs, dtype=torch.float)
        for i in range(imgs.shape[0]):
            single_image = imgs[i]
            image_pil = Image.fromarray(
                (single_image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            )

            boxes, _ = mtcnn.detect(image_pil)

            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box)
                    mask[i, :, y1:y2, x1:x2] = face_ratio

        return mask

    def __get_protect_both_swap_imgs(
        self, imgs1: torch.tensor, imgs2: torch.tensor, pert_imgs1: torch.tensor
    ) -> tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
        imgs1_identity = self._get_imgs_identity(imgs1)
        imgs1_src_swap = self.target(None, imgs2, imgs1_identity, None, True)

        pert_imgs1_identity = self._get_imgs_identity(pert_imgs1)
        pert_imgs1_src_swap = self.target(None, imgs2, pert_imgs1_identity, None, True)

        imgs2_identity = self._get_imgs_identity(imgs2)
        imgs1_tgt_swap = self.target(None, imgs1, imgs2_identity, None, True)

        pert_imgs1_tgt_swap = self.target(None, pert_imgs1, imgs2_identity, None, True)

        return imgs1_src_swap, pert_imgs1_src_swap, imgs1_tgt_swap, pert_imgs1_tgt_swap

    def __get_anchor_imgs_path(self) -> list[str]:
        all_imgs_path = sorted(os.listdir(self.anchorset_dir))
        all_imgs_path = [join(self.anchorset_dir, name) for name in all_imgs_path]

        return all_imgs_path

    def __find_best_anchor(
        self, imgs: torch.tensor, anchor_imgs: torch.tensor
    ) -> torch.tensor:
        imgs_ndarray = imgs.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255.0
        anchor_img_ndarray = (
            anchor_imgs.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255.0
        )
        best_anchors = []
        for i in range(imgs.shape[0]):
            distances = []
            for j in range(anchor_imgs.shape[0]):
                distance = self.effectiveness.get_distance(
                    imgs_ndarray[i], anchor_img_ndarray[j]
                )
                distances.append(distance)

            best_anchor_idx = 0
            for i, distance in enumerate(distances):
                if abs(distances[i] - 0.91906) < abs(
                    distances[best_anchor_idx] - 0.91906
                ):
                    best_anchor_idx = i

            best_anchors.append(anchor_imgs[best_anchor_idx])

        return torch.stack(best_anchors, dim=0)

    def pgd_both_sample(self) -> None:
        loss_weights = {"pert": 500, "identity": 1000, "latent": 0.5}
        loss_limits = {"latent": 15}
        self.logger.info(f"loss_weights: {loss_weights}, loss_limits: {loss_limits}")

        self.target.cuda().eval()
        l2_loss = nn.MSELoss().cuda()

        protect_path = [
            join(self.samples_dir, "zjl.jpg"),
            join(self.samples_dir, "6.jpg"),
            join(self.samples_dir, "hzxc.jpg"),
        ]

        protect_imgs = self._load_imgs(protect_path)
        x_imgs = protect_imgs.clone().detach()

        anchor_imgs_path = self.__get_anchor_imgs_path()
        anchor_imgs = self._load_imgs(anchor_imgs_path)
        anchor_imgs = self.__find_best_anchor(protect_imgs, anchor_imgs)
        anchor_identity = self._get_imgs_identity(anchor_imgs)
        epsilon = (
            self.args.pgd_epsilon
            * (torch.max(protect_imgs) - torch.min(protect_imgs))
            / 2
        )
        mask = self.__get_face_mask(x_imgs, self.args.face_mask_ratio)
        for epoch in range(self.args.epochs):
            x_imgs.requires_grad = True

            x_identity = self._get_imgs_identity(x_imgs)
            x_latent_code = self.target.netG.encoder(x_imgs)
            mimic_latent_code = self.target.netG.encoder(anchor_imgs)

            pert_diff_loss = l2_loss(x_imgs, protect_imgs.detach())
            identity_diff_loss = l2_loss(x_identity, anchor_identity.detach())
            latent_code_diff_loss = torch.clamp(
                l2_loss(x_latent_code, mimic_latent_code.detach()),
                0.0,
                loss_limits["latent"],
            )

            loss = (
                loss_weights["pert"] * pert_diff_loss
                + loss_weights["identity"] * identity_diff_loss
                - loss_weights["latent"] * latent_code_diff_loss
            )

            loss.backward(retain_graph=True)

            x_imgs = (
                x_imgs.clone().detach()
                - mask * epsilon * x_imgs.grad.sign().clone().detach()
            )
            x_imgs = torch.clamp(
                x_imgs,
                min=protect_imgs - self.args.pgd_limit,
                max=protect_imgs + self.args.pgd_limit,
            )
            self.logger.info(
                f"[Epoch {epoch:4}]loss: {loss:.5f}({loss_weights['pert'] * pert_diff_loss.item():.5f}, {loss_weights['identity'] * identity_diff_loss.item():.5f}, {loss_weights['latent'] * latent_code_diff_loss.item():.5f})"
            )

        other_path = [
            join(self.samples_dir, "zrf.jpg"),
            join(self.samples_dir, "zrf.jpg"),
            join(self.samples_dir, "zrf.jpg"),
        ]
        other_imgs = self._load_imgs(other_path)
        imgs1_src_swap, pert_imgs1_src_swap, imgs1_tgt_swap, pert_imgs1_tgt_swap = (
            self.__get_protect_both_swap_imgs(protect_imgs, other_imgs, x_imgs)
        )

        results = torch.cat(
            (
                protect_imgs,
                other_imgs,
                x_imgs,
                anchor_imgs,
                imgs1_src_swap,
                pert_imgs1_src_swap,
                imgs1_tgt_swap,
                pert_imgs1_tgt_swap,
            ),
            dim=0,
        )
        save_image(
            results,
            join(self.args.log_dir, "image", f"summary.png"),
            nrow=len(protect_path),
        )
        del results

        pert_utilities = self.utility.calculate_utility(protect_imgs, x_imgs)
        pert_as_src_swap_utilities = self.utility.calculate_utility(
            imgs1_src_swap, pert_imgs1_src_swap
        )
        pert_as_tgt_swap_utilities = self.utility.calculate_utility(
            imgs1_tgt_swap, pert_imgs1_tgt_swap
        )
        src_effectivenesses = self._calculate_effectiveness(
            protect_imgs,
            other_imgs,
            x_imgs,
            imgs1_src_swap,
            pert_imgs1_src_swap,
            anchor_imgs,
            True,
        )
        tgt_effectivenesses = self._calculate_effectiveness(
            other_imgs,
            protect_imgs,
            x_imgs,
            imgs1_tgt_swap,
            pert_imgs1_tgt_swap,
            anchor_imgs,
            True,
        )

        self.logger.info(
            f"pert utility(mse, psnr, ssim, lpips): ({pert_utilities['mse']:.3f}, {pert_utilities['psnr']:.3f}, {pert_utilities['ssim']:.3f}, {pert_utilities['lpips']:.3f}), pert as source swap utility(mse, psnr, ssim, lpips): ({pert_as_src_swap_utilities['mse']:.3f}, {pert_as_src_swap_utilities['psnr']:.3f}, {pert_as_src_swap_utilities['ssim']:.3f}, {pert_as_src_swap_utilities['lpips']:.3f}), pert as target swap utility(mse, psnr, ssim, lpips): ({pert_as_tgt_swap_utilities['mse']:.3f}, {pert_as_tgt_swap_utilities['psnr']:.3f}, {pert_as_tgt_swap_utilities['ssim']:.3f}, {pert_as_tgt_swap_utilities['lpips']:.3f}), pert as source effectivenesses(pert, swap, pert_swap, anchor): ({src_effectivenesses['pert']:.3f}, {src_effectivenesses['swap']:.3f}, {src_effectivenesses['pert_swap']:.3f}, {src_effectivenesses['anchor']:.3f}), pert as target effectivenesses(swap, pert_swap): ({tgt_effectivenesses['swap']:.3f}, {tgt_effectivenesses['pert_swap']:.3f})"
        )

    def pgd_both_metric(self) -> None:
        loss_weights = {"pert": 500, "identity": 1000, "latent": 0.5}
        loss_limits = {"latent": 15}
        self.logger.info(f"loss_weights: {loss_weights}, loss_limits: {loss_limits}")

        self.target.cuda().eval()
        l2_loss = nn.MSELoss().cuda()

        source_imgs_path, target_imgs_path = self._get_split_test_imgs_path()
        data = {
            "pert_utility": (0, 0, 0, 0),
            "pert_as_src_swap_utility": (0, 0, 0, 0),
            "pert_as_tgt_swap_utility": (0, 0, 0, 0),
            "pert_as_src_effectiveness": (0, 0, 0, 0),
            "pert_as_tgt_effectiveness": (0, 0, 0, 0),
        }

        anchor_imgs_path = self.__get_anchor_imgs_path()
        anchor_imgs = self._load_imgs(anchor_imgs_path)

        total_batch = (
            min(len(source_imgs_path), len(target_imgs_path)) // self.args.batch_size
        )
        for i in range(total_batch):
            iter_source_path = source_imgs_path[
                i * self.args.batch_size : (i + 1) * self.args.batch_size
            ]
            iter_target_path = target_imgs_path[
                i * self.args.batch_size : (i + 1) * self.args.batch_size
            ]

            source_imgs = self._load_imgs(iter_source_path)
            target_imgs = self._load_imgs(iter_target_path)
            best_anchor_imgs = self.__find_best_anchor(source_imgs, anchor_imgs)
            best_anchor_identity = self._get_imgs_identity(best_anchor_imgs)

            x_imgs = source_imgs.clone().detach()
            epsilon = (
                self.args.pgd_epsilon
                * (torch.max(source_imgs) - torch.min(source_imgs))
                / 2
            )
            mask = self.__get_face_mask(x_imgs)
            for epoch in range(self.args.epochs):
                x_imgs.requires_grad = True

                x_identity = self._get_imgs_identity(x_imgs)
                identity_diff_loss = l2_loss(x_identity, best_anchor_identity.detach())
                x_latent_code = self.target.netG.encoder(x_imgs)
                mimic_latent_code = self.target.netG.encoder(best_anchor_imgs)

                pert_diff_loss = l2_loss(x_imgs, source_imgs.clone().detach())
                identity_diff_loss = l2_loss(x_identity, best_anchor_identity.detach())
                latent_code_diff_loss = torch.clamp(
                    l2_loss(x_latent_code, mimic_latent_code.detach()),
                    0.0,
                    loss_limits["latent"],
                )

                loss = (
                    loss_weights["pert"] * pert_diff_loss
                    + loss_weights["identity"] * identity_diff_loss
                    - loss_weights["latent"] * latent_code_diff_loss
                )
                loss.backward(retain_graph=True)

                x_imgs = (
                    x_imgs.clone().detach()
                    - mask * epsilon * x_imgs.grad.sign().clone().detach()
                )
                x_imgs = torch.clamp(
                    x_imgs,
                    min=source_imgs - self.args.pgd_limit,
                    max=source_imgs + self.args.pgd_limit,
                )

                self.logger.info(
                    f"[Batch {i+1:4}/{total_batch:4}][Epoch {epoch:4}/{self.args.epochs:4}]loss: {loss:.5f}({loss_weights['pert'] * pert_diff_loss.item():.5f}, {loss_weights['identity'] * identity_diff_loss.item():.5f}, {loss_weights['latent'] * latent_code_diff_loss.item():.5f})"
                )

            imgs1_src_swap, pert_imgs1_src_swap, imgs1_tgt_swap, pert_imgs1_tgt_swap = (
                self.__get_protect_both_swap_imgs(source_imgs, target_imgs, x_imgs)
            )

            results = torch.cat(
                (
                    source_imgs,
                    target_imgs,
                    x_imgs,
                    imgs1_src_swap,
                    pert_imgs1_src_swap,
                    imgs1_tgt_swap,
                    pert_imgs1_tgt_swap,
                ),
                dim=0,
            )
            save_image(
                results,
                join(self.args.log_dir, "image", f"batch{i+1}.png"),
                nrow=len(source_imgs),
            )
            del results

            pert_utilities = self.utility.calculate_utility(source_imgs, x_imgs)
            pert_as_src_swap_utilities = self.utility.calculate_utility(
                imgs1_src_swap, pert_imgs1_src_swap
            )
            pert_as_tgt_swap_utilities = self.utility.calculate_utility(
                imgs1_tgt_swap, pert_imgs1_tgt_swap
            )
            src_effectivenesses = self._calculate_effectiveness(
                source_imgs,
                target_imgs,
                x_imgs,
                imgs1_src_swap,
                pert_imgs1_src_swap,
                best_anchor_imgs,
                True,
            )
            tgt_effectivenesses = self._calculate_effectiveness(
                target_imgs,
                source_imgs,
                x_imgs,
                imgs1_tgt_swap,
                pert_imgs1_tgt_swap,
                best_anchor_imgs,
                True,
            )

            data["pert_utility"] = tuple(
                x + y
                for x, y in zip(
                    data["pert_utility"],
                    (
                        pert_utilities["mse"],
                        pert_utilities["ssim"],
                        pert_utilities["psnr"],
                        pert_utilities["lpips"],
                    ),
                )
            )
            data["pert_as_src_swap_utility"] = tuple(
                x + y
                for x, y in zip(
                    data["pert_as_src_swap_utility"],
                    (
                        pert_as_src_swap_utilities["mse"],
                        pert_as_src_swap_utilities["ssim"],
                        pert_as_src_swap_utilities["psnr"],
                        pert_as_src_swap_utilities["lpips"],
                    ),
                )
            )
            data["pert_as_tgt_swap_utility"] = tuple(
                x + y
                for x, y in zip(
                    data["pert_as_tgt_swap_utility"],
                    (
                        pert_as_tgt_swap_utilities["mse"],
                        pert_as_tgt_swap_utilities["ssim"],
                        pert_as_tgt_swap_utilities["psnr"],
                        pert_as_tgt_swap_utilities["lpips"],
                    ),
                )
            )
            data["pert_as_src_effectiveness"] = tuple(
                x + y
                for x, y in zip(
                    data["pert_as_src_effectiveness"],
                    (
                        src_effectivenesses["pert"],
                        src_effectivenesses["swap"],
                        src_effectivenesses["pert_swap"],
                        src_effectivenesses["anchor"],
                    ),
                )
            )
            data["pert_as_tgt_effectiveness"] = tuple(
                x + y
                for x, y in zip(
                    data["pert_as_tgt_effectiveness"],
                    (
                        tgt_effectivenesses["pert"],
                        tgt_effectivenesses["swap"],
                        tgt_effectivenesses["pert_swap"],
                        tgt_effectivenesses["anchor"],
                    ),
                )
            )
            del imgs1_src_swap, pert_imgs1_src_swap, imgs1_tgt_swap, pert_imgs1_tgt_swap
            torch.cuda.empty_cache()

            self.logger.info(
                f"pert utility(mse, psnr, ssim, lpips): ({pert_utilities['mse']:.3f}, {pert_utilities['psnr']:.3f}, {pert_utilities['ssim']:.3f}, {pert_utilities['lpips']:.3f}), pert as source swap utility(mse, psnr, ssim, lpips): ({pert_as_src_swap_utilities['mse']:.3f}, {pert_as_src_swap_utilities['psnr']:.3f}, {pert_as_src_swap_utilities['ssim']:.3f}, {pert_as_src_swap_utilities['lpips']:.3f}), pert as target swap utility(mse, psnr, ssim, lpips): ({pert_as_tgt_swap_utilities['mse']:.3f}, {pert_as_tgt_swap_utilities['psnr']:.3f}, {pert_as_tgt_swap_utilities['ssim']:.3f}, {pert_as_tgt_swap_utilities['lpips']:.3f}), pert as src effectivenesses(pert, swap, pert_swap, anchor): ({src_effectivenesses['pert']:.3f}, {src_effectivenesses['swap']:.3f}, {src_effectivenesses['pert_swap']:.3f}, {src_effectivenesses['anchor']:.3f}), pert as tgt effectivenesses(swap, pert_swap): ({src_effectivenesses['swap']:.3f}, {src_effectivenesses['pert_swap']:.3f})"
            )

            self.logger.info(
                f"Average of {self.args.batch_size * (i + 1)} pictures: pert utility(mse, psnr, ssim, lpips): {tuple(f'{x / (i + 1):.3f}' for x in data['pert_utility'])}, pert as source swap utility(mse, psnr, ssim, lpips): {tuple(f'{x / (i + 1):.3f}' for x in data['pert_as_src_swap_utility'])}, pert as target swap utility(mse, psnr, ssim, lpips): {tuple(f'{x / (i + 1):.3f}' for x in data['pert_as_tgt_swap_utility'])}, pert as src effectiveness(pert, swap, pert swap, anchor): {tuple(f'{x / (i + 1):.3f}' for x in data['pert_as_src_effectiveness'])}, pert as tgt effectiveness(swap, pert swap): {data['pert_as_tgt_effectiveness'][1] / (i + 1):.3f}, {data['pert_as_tgt_effectiveness'][2] / (i + 1):.3f}"
            )

    def _calculate_utility(
        self, clean_imgs: torch.tensor, pert_imgs: torch.tensor
    ) -> float:
        clean_imgs_ndarray = clean_imgs.detach().cpu().numpy().transpose(0, 2, 3, 1)
        pert_imgs_ndarray = pert_imgs.detach().cpu().numpy().transpose(0, 2, 3, 1)
        mse, psnr, ssim = self.utility.compare(clean_imgs_ndarray, pert_imgs_ndarray)

        return mse, psnr, ssim

    def _calculate_distance(
        self,
        source_imgs: torch.tensor,
        target_imgs: torch.tensor,
        anchor_imgs: torch.tensor,
        pert_imgs_swap: torch.tensor,
    ) -> tuple[float, float, float]:
        source_imgs_ndarray = (
            source_imgs.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255.0
        )
        target_imgs_ndarray = (
            target_imgs.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255.0
        )
        anchor_imgs_ndarray = (
            anchor_imgs.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255.0
        )
        pert_imgs_swap_ndarray = (
            pert_imgs_swap.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255.0
        )

        _, source_to_pert_swap_dist = self.effectiveness.compare(
            source_imgs_ndarray, pert_imgs_swap_ndarray
        )
        _, target_to_pert_swap_dist = self.effectiveness.compare(
            target_imgs_ndarray, pert_imgs_swap_ndarray
        )
        _, anchor_to_pert_swap_dist = self.effectiveness.compare(
            anchor_imgs_ndarray, pert_imgs_swap_ndarray
        )

        return (
            source_to_pert_swap_dist,
            target_to_pert_swap_dist,
            anchor_to_pert_swap_dist,
        )

    def _calculate_effectiveness(
        self,
        source_imgs: torch.tensor,
        target_imgs: torch.tensor,
        pert_imgs: torch.tensor,
        clean_imgs_swap: torch.tensor,
        pert_imgs_swap: torch.tensor,
        anchor_imgs: torch.tensor = None,
    ):
        effectivenesses = {"pert": 0, "swap": 0, "pert_swap": 0, "anchor": 0}
        source_imgs_ndarray = (
            source_imgs.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255.0
        )
        pert_imgs_ndarray = (
            pert_imgs.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255.0
        )
        clean_imgs_swap_ndarray = (
            clean_imgs_swap.detach().cpu().numpy().transpose(0, 2, 3, 1)
        ) * 255.0
        pert_imgs_swap_ndarray = (
            pert_imgs_swap.detach().cpu().numpy().transpose(0, 2, 3, 1)
        ) * 255.0

        if target_imgs is None:
            pert, _ = self.effectiveness.compare(source_imgs_ndarray, pert_imgs_ndarray)
        else:
            target_imgs_ndarray = (
                target_imgs.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255.0
            )
            pert, _ = self.effectiveness.compare(target_imgs_ndarray, pert_imgs_ndarray)
        clean_swap, _ = self.effectiveness.compare(
            source_imgs_ndarray, clean_imgs_swap_ndarray
        )
        pert_swap, _ = self.effectiveness.compare(
            source_imgs_ndarray, pert_imgs_swap_ndarray
        )

        if anchor_imgs is not None:
            anchor_imgs_ndarray = (
                anchor_imgs.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255.0
            )
            anchor, _ = self.effectiveness.compare(
                anchor_imgs_ndarray, pert_imgs_swap_ndarray
            )
            return pert, clean_swap, pert_swap, anchor

        return pert, clean_swap, pert_swap

    def _calculate_effectiveness(
        self,
        source_imgs: torch.tensor,
        target_imgs: torch.tensor,
        pert_imgs: torch.tensor,
        imgs_swap: torch.tensor,
        pert_imgs_swap: torch.tensor,
        anchor_imgs: torch.tensor,
        protect_source: bool = True,
    ):
        effectivenesses = {"pert": 0, "swap": 0, "pert_swap": 0, "anchor": 0}

        source_imgs_ndarray = (
            source_imgs.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255.0
        )
        pert_imgs_ndarray = (
            pert_imgs.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255.0
        )
        imgs_swap_ndarray = (
            imgs_swap.detach().cpu().numpy().transpose(0, 2, 3, 1)
        ) * 255.0
        pert_imgs_swap_ndarray = (
            pert_imgs_swap.detach().cpu().numpy().transpose(0, 2, 3, 1)
        ) * 255.0

        swap, _ = self.effectiveness.compare(source_imgs_ndarray, imgs_swap_ndarray)
        effectivenesses["swap"] = swap
        pert_swap, _ = self.effectiveness.compare(
            source_imgs_ndarray, pert_imgs_swap_ndarray
        )
        effectivenesses["pert_swap"] = pert_swap

        if protect_source:
            pert, _ = self.effectiveness.compare(source_imgs_ndarray, pert_imgs_ndarray)
            effectivenesses["pert"] = pert
            if anchor_imgs is not None:
                anchor_imgs_ndarray = (
                    anchor_imgs.detach().cpu().numpy().transpose(0, 2, 3, 1)
                ) * 255.0
                anchor, _ = self.effectiveness.compare(
                    anchor_imgs_ndarray, pert_imgs_swap_ndarray
                )
                effectivenesses["anchor"] = anchor
        else:
            target_imgs_ndarray = (
                target_imgs.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255.0
            )
            pert, _ = self.effectiveness.compare(target_imgs_ndarray, pert_imgs_ndarray)
            effectivenesses["pert"] = pert

        return effectivenesses

    def pgd_source_metric(self, loss_weights=[1, 1]):
        self.logger.info(f"loss_weights: {loss_weights}")

        self.target.cuda().eval()
        l2_loss = nn.MSELoss().cuda()

        source_imgs_path, target_imgs_path = self._get_split_test_imgs_path()
        data = {
            "pert_mse": [],
            "pert_psnr": [],
            "pert_ssim": [],
            "pert_swap_mse": [],
            "pert_swap_psnr": [],
            "pert_swap_ssim": [],
            "pert_effectiveness": [],
            "swap_effectiveness": [],
            "pert_swap_effectiveness": [],
            "anchor_effectiveness": [],
        }

        total_batch = (
            min(len(source_imgs_path), len(target_imgs_path)) // self.args.batch_size
        )

        mimic_img = self._load_imgs([join(self.args.data_dir, self.args.pgd_mimic)])
        mimic_img_expand = mimic_img.repeat(self.args.batch_size, 1, 1, 1)
        mimic_identity_expand = self._get_imgs_identity(mimic_img_expand)
        for i in range(total_batch):
            iter_source_path = source_imgs_path[
                i * self.args.batch_size : (i + 1) * self.args.batch_size
            ]
            iter_target_path = target_imgs_path[
                i * self.args.batch_size : (i + 1) * self.args.batch_size
            ]

            source_imgs = self._load_imgs(iter_source_path)
            target_imgs = self._load_imgs(iter_target_path)
            source_identity = self._get_imgs_identity(source_imgs)
            swap_imgs = self.target(None, target_imgs, source_identity, None, True)

            x_imgs = source_imgs.clone().detach()
            epsilon = (
                self.args.pgd_epsilon
                * (torch.max(source_imgs) - torch.min(source_imgs))
                / 2
            )

            for epoch in range(self.args.epochs):
                x_imgs.requires_grad = True

                x_identity = torch.empty(self.args.batch_size, 512).cuda()
                for j in range(self.args.batch_size):
                    identity = self._get_imgs_identity(x_imgs[j].unsqueeze(0))
                    x_identity[j] = identity[0]
                    del identity

                pert_diff_loss = l2_loss(x_imgs, source_imgs.detach())
                identity_diff_loss = l2_loss(x_identity, mimic_identity_expand.detach())

                loss = (
                    loss_weights[0] * pert_diff_loss
                    + loss_weights[1] * identity_diff_loss
                )
                loss.backward()

                x_imgs = (
                    x_imgs.clone().detach()
                    - epsilon * x_imgs.grad.sign().clone().detach()
                )
                x_imgs = torch.clamp(
                    x_imgs,
                    min=source_imgs - self.args.pgd_limit,
                    max=source_imgs + self.args.pgd_limit,
                )

                self.logger.info(
                    f"Iter {i:5}/{total_batch:5}[Epoch {epoch+1:3}/{self.args.epochs:3}]loss: {loss:.5f}({loss_weights[0] * pert_diff_loss.item():.5f}, {loss_weights[1] * identity_diff_loss.item():.5f})"
                )

            x_identity = self._get_imgs_identity(x_imgs)
            x_swap_img = self.target(None, target_imgs, x_identity, None, True)

            pert_mse, pert_psnr, pert_ssim = self._calculate_utility(
                source_imgs, x_imgs
            )
            data["pert_mse"].append(pert_mse)
            data["pert_psnr"].append(pert_psnr)
            data["pert_ssim"].append(pert_ssim)

            pert_swap_mse, pert_swap_psnr, pert_swap_ssim = self._calculate_utility(
                swap_imgs, x_swap_img
            )
            data["pert_swap_mse"].append(pert_swap_mse)
            data["pert_swap_psnr"].append(pert_swap_psnr)
            data["pert_swap_ssim"].append(pert_swap_ssim)

            (
                pert_effectiveness,
                swap_effectiveness,
                pert_swap_effectiveness,
                anchor_effectiveness,
            ) = self._calculate_effectiveness(
                source_imgs, None, x_imgs, swap_imgs, x_swap_img, mimic_img_expand
            )
            data["pert_effectiveness"].append(pert_effectiveness)
            data["swap_effectiveness"].append(swap_effectiveness)
            data["pert_swap_effectiveness"].append(pert_swap_effectiveness)
            data["anchor_effectiveness"].append(anchor_effectiveness)

            if i % self.args.log_interval == 0:
                results = torch.cat(
                    (source_imgs, target_imgs, swap_imgs, x_imgs, x_swap_img), dim=0
                )
                save_path = join(self.args.log_dir, "image", f"pgd_source_{i}.png")
                save_image(results, save_path, nrow=self.args.batch_size)
                del results

            del source_imgs, target_imgs, source_identity, swap_imgs
            del x_imgs, x_identity, x_swap_img
            torch.cuda.empty_cache()

            self.logger.info(
                f"Iter {i:5}/{total_batch:5}, pert utility(mse, psnr, ssim): {pert_mse:.3f}, {pert_psnr:.3f}, {pert_ssim:.3f}, pert swap utility(mse, psnr, ssim): {pert_swap_mse:.3f}, {pert_swap_psnr:.3f}, {pert_swap_ssim:.3f}, effectiveness (pert, clean swap, pert swap, anchor): {pert_effectiveness:.3f}, {swap_effectiveness:.3f}, {pert_swap_effectiveness:.3f}, {anchor_effectiveness:.3f}"
            )

            self.logger.info(
                f"Average of {self.args.batch_size * (i + 1)} pictures: pert utility(mse, psnr, ssim): {sum(data['pert_mse'])/len(data['pert_mse']):.3f} {sum(data['pert_psnr'])/len(data['pert_psnr']):.3f} {sum(data['pert_ssim'])/len(data['pert_ssim']):.3f}, pert swap utility(mse, psnr, ssim): {sum(data['pert_swap_mse'])/len(data['pert_swap_mse']):.3f} {sum(data['pert_swap_psnr'])/len(data['pert_swap_psnr']):.3f} {sum(data['pert_swap_ssim'])/len(data['pert_swap_ssim']):.3f}, effectiveness(pert, swap, pert swap, anchor): {sum(data['pert_effectiveness'])/len(data['pert_effectiveness']):.3f}, {sum(data['swap_effectiveness'])/len(data['swap_effectiveness']):.3f}, {sum(data['pert_swap_effectiveness'])/len(data['pert_swap_effectiveness']):.3f}, {sum(data['anchor_effectiveness'])/len(data['anchor_effectiveness']):.3f}"
            )

    def pgd_source_distance(self, loss_weights=[1, 1]):
        self.logger.info(f"loss_weights: {loss_weights}")

        self.target.cuda().eval()
        l2_loss = nn.MSELoss().cuda()

        source_imgs_path, target_imgs_path = self._get_split_test_imgs_path()
        total_batch = (
            min(len(source_imgs_path), len(target_imgs_path)) // self.args.batch_size
        )
        mimic_img = self._load_imgs([join(self.args.data_dir, self.args.pgd_mimic)])
        mimic_img_expand = mimic_img.repeat(self.args.batch_size, 1, 1, 1)
        mimic_identity_expand = self._get_imgs_identity(mimic_img_expand)

        distance_between_swap_to = {"source": [], "target": [], "anchor": []}
        distance_between_pert_swap_to = {"source": [], "target": [], "anchor": []}
        for i in range(total_batch):
            iter_source_path = source_imgs_path[
                i * self.args.batch_size : (i + 1) * self.args.batch_size
            ]
            iter_target_path = target_imgs_path[
                i * self.args.batch_size : (i + 1) * self.args.batch_size
            ]

            source_imgs = self._load_imgs(iter_source_path)
            target_imgs = self._load_imgs(iter_target_path)
            source_identity = self._get_imgs_identity(source_imgs)
            swap_imgs = self.target(None, target_imgs, source_identity, None, True)

            x_imgs = source_imgs.clone().detach()
            epsilon = (
                self.args.pgd_epsilon
                * (torch.max(source_imgs) - torch.min(source_imgs))
                / 2
            )

            for epoch in range(self.args.epochs):
                x_imgs.requires_grad = True

                x_identity = torch.empty(self.args.batch_size, 512).cuda()
                for j in range(self.args.batch_size):
                    identity = self._get_imgs_identity(x_imgs[j].unsqueeze(0))
                    x_identity[j] = identity[0]

                pert_diff_loss = l2_loss(x_imgs, source_imgs.detach())
                identity_diff_loss = l2_loss(x_identity, mimic_identity_expand.detach())

                loss = (
                    loss_weights[0] * pert_diff_loss
                    + loss_weights[1] * identity_diff_loss
                )
                loss.backward()

                x_imgs = (
                    x_imgs.clone().detach()
                    - epsilon * x_imgs.grad.sign().clone().detach()
                )
                x_imgs = torch.clamp(
                    x_imgs,
                    min=source_imgs - self.args.pgd_limit,
                    max=source_imgs + self.args.pgd_limit,
                )

                self.logger.info(
                    f"Iter {i:5}/{total_batch:5}[Epoch {epoch+1:3}/{self.args.epochs:3}]loss: {loss:.5f}({loss_weights[0] * pert_diff_loss.item():.5f}, {loss_weights[1] * identity_diff_loss.item():.5f})"
                )

            x_identity = self._get_imgs_identity(x_imgs)
            x_swap_img = self.target(None, target_imgs, x_identity, None, True)

            (
                source_to_swap_dist,
                target_to_swap_dist,
                anchor_to_swap_dist,
            ) = self._calculate_distance(
                source_imgs, target_imgs, mimic_img_expand, swap_imgs
            )
            distance_between_swap_to["source"].append(source_to_swap_dist)
            distance_between_swap_to["target"].append(target_to_swap_dist)
            distance_between_swap_to["anchor"].append(anchor_to_swap_dist)
            (
                source_to_pert_swap_dist,
                target_to_pert_swap_dist,
                anchor_to_pert_swap_dist,
            ) = self._calculate_distance(
                source_imgs, target_imgs, mimic_img_expand, x_swap_img
            )
            distance_between_pert_swap_to["source"].append(source_to_pert_swap_dist)
            distance_between_pert_swap_to["target"].append(target_to_pert_swap_dist)
            distance_between_pert_swap_to["anchor"].append(anchor_to_pert_swap_dist)

            if i % self.args.log_interval == 0:
                results = torch.cat(
                    (source_imgs, target_imgs, swap_imgs, x_imgs, x_swap_img), dim=0
                )
                save_path = join(self.args.log_dir, "image", f"pgd_source_{i}.png")
                save_image(results, save_path, nrow=self.args.batch_size)
                del results

            del x_imgs, x_identity, x_swap_img
            torch.cuda.empty_cache()

            self.logger.info(
                f"Iter {i:5}/{total_batch:5}, Distance (swap to source, target, anchor): {source_to_swap_dist:.5f}, {target_to_swap_dist:.5f}, {anchor_to_swap_dist:.5f}, (pert swap to source, target, anchor): {source_to_pert_swap_dist:.5f}, {target_to_pert_swap_dist:.5f}, {anchor_to_pert_swap_dist:.5f}"
            )

            self.logger.info(
                f"Average distance of {self.args.batch_size * (i + 1)} pictures (swap to source, target, anchor): {sum(distance_between_swap_to['source'])/len(distance_between_swap_to['source']):.5f}, {sum(distance_between_swap_to['target'])/len(distance_between_swap_to['target']):.5f}, {sum(distance_between_swap_to['anchor'])/len(distance_between_swap_to['anchor']):.5f}, (pert swap to source, target, anchor): {sum(distance_between_pert_swap_to['source'])/len(distance_between_pert_swap_to['source']):.5f}, {sum(distance_between_pert_swap_to['target'])/len(distance_between_pert_swap_to['target']):.5f}, {sum(distance_between_pert_swap_to['anchor'])/len(distance_between_pert_swap_to['anchor']):.5f}"
            )

    def pgd_target_metric(self, loss_weights=[5000, 1]):
        self.logger.info(f"loss_weights: {loss_weights}")

        self.target.cuda().eval()
        l2_loss = nn.MSELoss().cuda()

        source_imgs_path, target_imgs_path = self._get_split_test_imgs_path()
        data = {
            "pert_mse": [],
            "pert_psnr": [],
            "pert_ssim": [],
            "pert_swap_mse": [],
            "pert_swap_psnr": [],
            "pert_swap_ssim": [],
            "pert_effectiveness": [],
            "swap_effectiveness": [],
            "pert_swap_effectiveness": [],
        }

        total_batch = (
            min(len(source_imgs_path), len(target_imgs_path)) // self.args.batch_size
        )
        mimic_img = self._load_imgs([join(self.args.data_dir, self.args.pgd_mimic)])
        mimic_img_expand = mimic_img.repeat(self.args.batch_size, 1, 1, 1)
        mimic_latent_code = self.target.netG.encoder(mimic_img_expand)
        for i in range(total_batch):
            iter_source_path = source_imgs_path[
                i * self.args.batch_size : (i + 1) * self.args.batch_size
            ]
            iter_target_path = target_imgs_path[
                i * self.args.batch_size : (i + 1) * self.args.batch_size
            ]

            source_imgs = self._load_imgs(iter_source_path)
            target_imgs = self._load_imgs(iter_target_path)
            source_identity = self._get_imgs_identity(source_imgs)
            swap_imgs = self.target(None, target_imgs, source_identity, None, True)

            x_imgs = target_imgs.clone().detach()
            epsilon = (
                self.args.pgd_epsilon
                * (torch.max(target_imgs) - torch.min(target_imgs))
                / 2
            )
            for epoch in range(self.args.epochs):
                x_imgs.requires_grad = True

                x_latent_code = self.target.netG.encoder(x_imgs)
                pert_diff_loss = l2_loss(x_imgs, target_imgs.detach())
                latent_code_diff_loss = l2_loss(
                    x_latent_code, mimic_latent_code.detach()
                )
                loss = (
                    loss_weights[0] * pert_diff_loss
                    - loss_weights[1] * latent_code_diff_loss
                )

                loss.backward(retain_graph=True)

                x_imgs = (
                    x_imgs.clone().detach()
                    - epsilon * x_imgs.grad.sign().clone().detach()
                )
                x_imgs = torch.clamp(
                    x_imgs,
                    min=target_imgs - self.args.pgd_limit,
                    max=target_imgs + self.args.pgd_limit,
                )

                self.logger.info(
                    f"Iter {i:5}/{total_batch:5}[Epoch {epoch+1:3}/{self.args.epochs:3}]loss: {loss:.5f}({loss_weights[0] * pert_diff_loss.item():.5f}, {loss_weights[1] * latent_code_diff_loss.item():.5f})"
                )

            x_swap_img = self.target(None, x_imgs, source_identity, None, True)

            pert_mse, pert_psnr, pert_ssim = self._calculate_utility(
                target_imgs, x_imgs
            )
            data["pert_mse"].append(pert_mse)
            data["pert_psnr"].append(pert_psnr)
            data["pert_ssim"].append(pert_ssim)

            pert_swap_mse, pert_swap_psnr, pert_swap_ssim = self._calculate_utility(
                swap_imgs, x_swap_img
            )
            data["pert_swap_mse"].append(pert_swap_mse)
            data["pert_swap_psnr"].append(pert_swap_psnr)
            data["pert_swap_ssim"].append(pert_swap_ssim)

            (pert_effectiveness, swap_effectiveness, pert_swap_effectiveness) = (
                self._calculate_effectiveness(
                    source_imgs, target_imgs, x_imgs, swap_imgs, x_swap_img
                )
            )
            data["pert_effectiveness"].append(pert_effectiveness)
            data["swap_effectiveness"].append(swap_effectiveness)
            data["pert_swap_effectiveness"].append(pert_swap_effectiveness)

            if i % self.args.log_interval == 0:
                results = torch.cat(
                    (source_imgs, target_imgs, swap_imgs, x_imgs, x_swap_img), dim=0
                )
                save_path = join(self.args.log_dir, "image", f"pgd_target_{i}.png")
                save_image(results, save_path, nrow=self.args.batch_size)
                del results

            del x_imgs, x_swap_img
            torch.cuda.empty_cache()

            self.logger.info(
                f"Iter {i:5}/{total_batch:5}, pert utility(mse, psnr, ssim): {pert_mse:.3f} {pert_psnr:.3f} {pert_ssim:.3f}, pert swap utility(mse, psnr, ssim): {pert_swap_mse:.3f} {pert_swap_psnr:.3f} {pert_swap_ssim:.3f}, effectiveness (pert, clean swap, pert swap): {pert_effectiveness:.3f}, {swap_effectiveness:.3f}, {pert_swap_effectiveness:.3f}"
            )

            self.logger.info(
                f"Average of {self.args.batch_size * (i + 1)} pictures: pert utility(mse, psnr, ssim): {sum(data['pert_mse'])/len(data['pert_mse']):.3f} {sum(data['pert_psnr'])/len(data['pert_psnr']):.3f} {sum(data['pert_ssim'])/len(data['pert_ssim']):.3f}, pert swap utility(mse, psnr, ssim): {sum(data['pert_swap_mse'])/len(data['pert_swap_mse']):.3f} {sum(data['pert_swap_psnr'])/len(data['pert_swap_psnr']):.3f} {sum(data['pert_swap_ssim'])/len(data['pert_swap_ssim']):.3f}, effectiveness(pert, swap, pert swap): {sum(data['pert_effectiveness'])/len(data['pert_effectiveness']):.3f}, {sum(data['swap_effectiveness'])/len(data['swap_effectiveness']):.3f}, {sum(data['pert_swap_effectiveness'])/len(data['pert_swap_effectiveness']):.3f}"
            )

    def _get_all_imgs_path(self, train_set: bool = True) -> list[str]:
        set_to_load = self.trainset_dir if train_set else self.testset_dir
        all_people = sorted(os.listdir(set_to_load))
        all_imgs_path = []
        for people in all_people:
            people_dir = join(set_to_load, people)
            all_imgs_name = sorted(os.listdir(people_dir))
            all_imgs_path.extend(
                [join(set_to_load, people, name) for name in all_imgs_name]
            )

        self.logger.info(
            f"Collect {len(all_imgs_path)} images for GAN {'training' if train_set else 'test'}"
        )
        return all_imgs_path

    def _get_average_identity(self) -> torch.tensor:
        mean_identity_path = join("checkpoints", "average_identity.pt")
        if os.path.exists(mean_identity_path):
            mean_identity = torch.load(mean_identity_path).cuda()
        else:
            train_imgs_path = self._get_all_imgs_path(train_set=True)
            identity_sum = None
            for i in tqdm(range(len(train_imgs_path) // self.args.batch_size)):
                imgs = self._load_imgs(
                    train_imgs_path[
                        i * self.args.batch_size : (i + 1) * self.args.batch_size
                    ]
                )
                identity = self._get_imgs_identity(imgs)
                identity = identity.detach()
                if identity_sum is None:
                    identity_sum = identity
                else:
                    identity_sum += identity
                del identity

            mean_identity = identity_sum / (
                len(train_imgs_path) // self.args.batch_size
            )
            mean_identity = torch.mean(mean_identity, dim=0)
            mean_identity = mean_identity.unsqueeze(0)
            print(mean_identity, mean_identity.shape)

            torch.save(mean_identity, mean_identity_path)

        return mean_identity

    def gan_source(self):
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
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_G, self.args.epochs)

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

        average_identity = self._get_average_identity().expand(
            self.args.batch_size, 512
        )
        for epoch in range(self.args.epochs):
            self.GAN_G.cuda().train()
            src_imgs_path = random.sample(train_imgs_path, self.args.batch_size)
            tgt_imgs_path = random.sample(train_imgs_path, self.args.batch_size)

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
            identity_mimic_loss = l2_loss(
                flatten(pert_src_identity), flatten(average_identity)
            )

            G_loss = (
                self.gan_src_loss_weights[0] * pert_diff_loss
                + self.gan_src_loss_weights[1] * swap_diff_loss
                + self.gan_src_loss_weights[2] * identity_diff_loss
                + self.gan_src_loss_weights[3] * identity_mimic_loss
            )
            G_loss.backward()
            optimizer_G.step()
            scheduler.step()

            self.logger.info(
                f"[Epoch {epoch:6}]loss: {G_loss:8.5f}({self.gan_src_loss_weights[0] * pert_diff_loss.item():.5f}, {self.gan_src_loss_weights[1] * swap_diff_loss.item():.5f}, {self.gan_src_loss_weights[2] * identity_diff_loss.item():.5f}, {self.gan_src_loss_weights[3] * identity_mimic_loss.item():.5f})({pert_diff_loss.item():.5f}, {swap_diff_loss.item():.5f}, {identity_diff_loss.item():.5f}, {identity_mimic_loss.item():.5f})"
            )

            if epoch % self.args.log_interval == 0:
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

    def gan_target(self):
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
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_G, self.args.epochs)

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
        for epoch in range(self.args.epochs):
            self.GAN_G.cuda().train()
            src_imgs_path = random.sample(train_imgs_path, self.args.batch_size)
            tgt_imgs_path = random.sample(train_imgs_path, self.args.batch_size)

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

            if epoch % self.args.log_interval == 0:
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
        all_people = sorted(os.listdir(self.testset_dir))
        random.shuffle(all_people)

        source_people = all_people[: int(len(all_people) / 2)]
        target_people = all_people[int(len(all_people) / 2) :]

        source_imgs_path = []
        for people in source_people:
            people_dir = join(self.testset_dir, people)
            people_imgs_name = sorted(os.listdir(people_dir))
            source_imgs_path.extend(
                [join(self.testset_dir, people, name) for name in people_imgs_name]
            )

        target_imgs_path = []
        for people in target_people:
            people_dir = join(self.testset_dir, people)
            people_imgs_name = sorted(os.listdir(people_dir))
            target_imgs_path.extend(
                [join(self.testset_dir, people, name) for name in people_imgs_name]
            )

        return source_imgs_path, target_imgs_path

    def __save_gan_samples(self, imgs: list[torch.tensor]) -> None:
        img_names = ["source", "target", "swap", "pert", "pert_swap"]

        for i, name in enumerate(img_names):
            for j in range(imgs[i].shape[0]):
                save_image(
                    imgs[i][j], join(self.args.log_dir, "image", f"{name}_{j}.png")
                )

        results = torch.cat(imgs, dim=0)
        save_image(
            results,
            join(self.args.log_dir, "image", f"summary.png"),
            nrow=imgs[0].shape[0],
        )
        del results

    def gan_source_sample(self):
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

        pert_source_imgs = self.GAN_G(source_imgs)
        pert_source_identity = self._get_imgs_identity(pert_source_imgs)
        pert_swap_imgs = self.target(
            None, target_imgs, pert_source_identity, None, True
        )

        self.__save_gan_samples(
            [source_imgs, target_imgs, swap_imgs, pert_source_imgs, pert_swap_imgs]
        )

        pert_mse, pert_psnr, pert_ssim = self._calculate_utility(
            source_imgs, pert_source_imgs
        )

        pert_swap_mse, pert_swap_psnr, pert_swap_ssim = self._calculate_utility(
            swap_imgs, pert_swap_imgs
        )

        (
            pert_effectiveness,
            swap_effectiveness,
            pert_swap_effectiveness,
        ) = self._calculate_effectiveness(
            source_imgs, None, pert_source_imgs, swap_imgs, pert_swap_imgs
        )

        del pert_source_imgs, pert_source_identity, pert_swap_imgs
        torch.cuda.empty_cache()

        self.logger.info(
            f"pert utility(mse, psnr, ssim): {pert_mse:.3f} {pert_psnr:.3f} {pert_ssim:.3f}, pert swap utility(mse, psnr, ssim): {pert_swap_mse:.3f} {pert_swap_psnr:.3f} {pert_swap_ssim:.3f}, effectiveness (pert, clean swap, pert swap): {pert_effectiveness:.3f}, {swap_effectiveness:.3f}, {pert_swap_effectiveness:.3f}"
        )

    def gan_target_sample(self):
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

        pert_target_imgs = self.GAN_G(target_imgs)
        pert_swap_imgs = self.target(
            None, pert_target_imgs, source_identity, None, True
        )

        self.__save_gan_samples(
            [source_imgs, target_imgs, swap_imgs, pert_target_imgs, pert_swap_imgs]
        )

        pert_mse, pert_psnr, pert_ssim = self._calculate_utility(
            target_imgs, pert_target_imgs
        )

        pert_swap_mse, pert_swap_psnr, pert_swap_ssim = self._calculate_utility(
            swap_imgs, pert_swap_imgs
        )

        (
            pert_effectiveness,
            swap_effectiveness,
            pert_swap_effectiveness,
        ) = self._calculate_effectiveness(
            source_imgs, target_imgs, pert_target_imgs, swap_imgs, pert_swap_imgs
        )

        self.logger.info(
            f"pert utility(mse, psnr, ssim): {pert_mse:.3f} {pert_psnr:.3f} {pert_ssim:.3f}, pert swap utility(mse, psnr, ssim): {pert_swap_mse:.3f} {pert_swap_psnr:.3f} {pert_swap_ssim:.3f}, effectiveness (pert, clean swap, pert swap): {pert_effectiveness:.3f}, {swap_effectiveness:.3f}, {pert_swap_effectiveness:.3f}"
        )

    def gan_source_metric(self):
        model_path = join("checkpoints", self.args.gan_test_models)
        self.GAN_G.load_state_dict(torch.load(model_path)["GAN_G_state_dict"])

        self.target.cuda().eval()
        self.GAN_G.cuda().eval()

        source_imgs_path, target_imgs_path = self._get_split_test_imgs_path()
        data = {
            "pert_mse": [],
            "pert_psnr": [],
            "pert_ssim": [],
            "pert_swap_mse": [],
            "pert_swap_psnr": [],
            "pert_swap_ssim": [],
            "pert_effectiveness": [],
            "swap_effectiveness": [],
            "pert_swap_effectiveness": [],
        }
        total_batch = (
            min(len(source_imgs_path), len(target_imgs_path)) // self.args.batch_size
        )
        for i in range(total_batch):
            iter_source_path = source_imgs_path[
                i * self.args.batch_size : (i + 1) * self.args.batch_size
            ]
            iter_target_path = target_imgs_path[
                i * self.args.batch_size : (i + 1) * self.args.batch_size
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

            pert_mse, pert_psnr, pert_ssim = self._calculate_utility(
                source_imgs, pert_source_imgs
            )
            data["pert_mse"].append(pert_mse)
            data["pert_psnr"].append(pert_psnr)
            data["pert_ssim"].append(pert_ssim)

            pert_swap_mse, pert_swap_psnr, pert_swap_ssim = self._calculate_utility(
                swap_imgs, pert_swap_imgs
            )
            data["pert_swap_mse"].append(pert_swap_mse)
            data["pert_swap_psnr"].append(pert_swap_psnr)
            data["pert_swap_ssim"].append(pert_swap_ssim)

            (
                pert_effectiveness,
                swap_effectiveness,
                pert_swap_effectiveness,
            ) = self._calculate_effectiveness(
                source_imgs, None, pert_source_imgs, swap_imgs, pert_swap_imgs
            )
            data["pert_effectiveness"].append(pert_effectiveness)
            data["swap_effectiveness"].append(swap_effectiveness)
            data["pert_swap_effectiveness"].append(pert_swap_effectiveness)

            if i % self.args.log_interval == 0:
                results = torch.cat(
                    (
                        source_imgs,
                        target_imgs,
                        swap_imgs,
                        pert_source_imgs,
                        pert_swap_imgs,
                    ),
                    dim=0,
                )
                save_path = join(self.args.log_dir, "image", f"gan_source_{i}.png")
                save_image(results, save_path, nrow=self.args.batch_size)
                del results

            del pert_source_imgs, pert_source_identity, pert_swap_imgs
            torch.cuda.empty_cache()

            self.logger.info(
                f"Iter {i:5}/{total_batch:5}, pert utility(mse, psnr, ssim): {pert_mse:.3f} {pert_psnr:.3f} {pert_ssim:.3f}, pert swap utility(mse, psnr, ssim): {pert_swap_mse:.3f} {pert_swap_psnr:.3f} {pert_swap_ssim:.3f}, effectiveness (pert, clean swap, pert swap): {pert_effectiveness:.3f}, {swap_effectiveness:.3f}, {pert_swap_effectiveness:.3f}"
            )

            self.logger.info(
                f"Average of {self.args.batch_size * (i+1)} pictures: pert utility(mse, psnr, ssim): {sum(data['pert_mse'])/len(data['pert_mse']):.3f} {sum(data['pert_psnr'])/len(data['pert_psnr']):.3f} {sum(data['pert_ssim'])/len(data['pert_ssim']):.3f}, pert swap utility(mse, psnr, ssim): {sum(data['pert_swap_mse'])/len(data['pert_swap_mse']):.3f} {sum(data['pert_swap_psnr'])/len(data['pert_swap_psnr']):.3f} {sum(data['pert_swap_ssim'])/len(data['pert_swap_ssim']):.3f}, effectiveness(pert, swap, pert swap, anchor): {sum(data['pert_effectiveness'])/len(data['pert_effectiveness']):.3f}, {sum(data['swap_effectiveness'])/len(data['swap_effectiveness']):.3f}, {sum(data['pert_swap_effectiveness'])/len(data['pert_swap_effectiveness']):.3f}"
            )

    def gan_target_metric(self):
        model_path = join("checkpoints", self.args.gan_test_models)
        self.GAN_G.load_state_dict(torch.load(model_path)["GAN_G_state_dict"])

        self.target.cuda().eval()
        self.GAN_G.cuda().eval()

        source_imgs_path, target_imgs_path = self._get_split_test_imgs_path()
        data = {
            "pert_mse": [],
            "pert_psnr": [],
            "pert_ssim": [],
            "pert_swap_mse": [],
            "pert_swap_psnr": [],
            "pert_swap_ssim": [],
            "pert_effectiveness": [],
            "swap_effectiveness": [],
            "pert_swap_effectiveness": [],
        }
        total_batch = (
            min(len(source_imgs_path), len(target_imgs_path)) // self.args.batch_size
        )
        for i in range(total_batch):
            iter_source_path = source_imgs_path[
                i * self.args.batch_size : (i + 1) * self.args.batch_size
            ]
            iter_target_path = target_imgs_path[
                i * self.args.batch_size : (i + 1) * self.args.batch_size
            ]

            source_imgs = self._load_imgs(iter_source_path)
            target_imgs = self._load_imgs(iter_target_path)
            source_identity = self._get_imgs_identity(source_imgs)
            swap_imgs = self.target(None, target_imgs, source_identity, None, True)

            pert_target_imgs = self.GAN_G(target_imgs)
            pert_swap_imgs = self.target(
                None, pert_target_imgs, source_identity, None, True
            )

            pert_mse, pert_psnr, pert_ssim = self._calculate_utility(
                target_imgs, pert_target_imgs
            )
            data["pert_mse"].append(pert_mse)
            data["pert_psnr"].append(pert_psnr)
            data["pert_ssim"].append(pert_ssim)

            pert_swap_mse, pert_swap_psnr, pert_swap_ssim = self._calculate_utility(
                swap_imgs, pert_swap_imgs
            )
            data["pert_swap_mse"].append(pert_swap_mse)
            data["pert_swap_psnr"].append(pert_swap_psnr)
            data["pert_swap_ssim"].append(pert_swap_ssim)

            pert_effectiveness, swap_effectiveness, pert_swap_effectiveness = (
                self._calculate_effectiveness(
                    source_imgs,
                    target_imgs,
                    pert_target_imgs,
                    swap_imgs,
                    pert_swap_imgs,
                )
            )
            data["pert_effectiveness"].append(pert_effectiveness)
            data["swap_effectiveness"].append(swap_effectiveness)
            data["pert_swap_effectiveness"].append(pert_swap_effectiveness)

            if i % self.args.log_interval == 0:
                results = torch.cat(
                    (
                        source_imgs,
                        target_imgs,
                        swap_imgs,
                        pert_target_imgs,
                        pert_swap_imgs,
                    ),
                    dim=0,
                )
                save_path = join(self.args.log_dir, "image", f"gan_target_{i}.png")
                save_image(results, save_path, nrow=self.args.batch_size)
                del results

            del pert_target_imgs, pert_swap_imgs

            self.logger.info(
                f"Iter {i:5}/{total_batch:5}, pert utility(mse, psnr, ssim): {pert_mse:.3f} {pert_psnr:.3f} {pert_ssim:.3f}, pert swap utility(mse, psnr, ssim): {pert_swap_mse:.3f} {pert_swap_psnr:.3f} {pert_swap_ssim:.3f}, effectiveness (pert, clean swap, pert swap): {pert_effectiveness:.3f}, {swap_effectiveness:.3f}, {pert_swap_effectiveness:.3f}"
            )

            self.logger.info(
                f"Average of {self.args.batch_size * (i+1)} pictures: pert utility(mse, psnr, ssim): {sum(data['pert_mse'])/len(data['pert_mse']):.3f} {sum(data['pert_psnr'])/len(data['pert_psnr']):.3f} {sum(data['pert_ssim'])/len(data['pert_ssim']):.3f}, pert swap utility(mse, psnr, ssim): {sum(data['pert_swap_mse'])/len(data['pert_swap_mse']):.3f} {sum(data['pert_swap_psnr'])/len(data['pert_swap_psnr']):.3f} {sum(data['pert_swap_ssim'])/len(data['pert_swap_ssim']):.3f}, effectiveness(pert, swap, pert swap, anchor): {sum(data['pert_effectiveness'])/len(data['pert_effectiveness']):.3f}, {sum(data['swap_effectiveness'])/len(data['swap_effectiveness']):.3f}, {sum(data['pert_swap_effectiveness'])/len(data['pert_swap_effectiveness']):.3f}"
            )

    def __gauss_noise(
        self, pert: torch.tensor, gauss_mean: float, gauss_std: float
    ) -> torch.tensor:
        gauss_noise = gauss_mean + gauss_std * torch.randn(pert.shape).cuda()
        noise_pert = pert + gauss_noise

        return noise_pert

    def __gauss_kernel(self, size: int, sigma: float):
        coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2.0
        grid = coords.repeat(size).view(size, size)
        kernel = torch.exp(-0.5 * (grid**2 + grid.t() ** 2) / sigma**2)
        kernel = kernel / kernel.sum()

        return kernel

    def __gauss_blur(self, pert: torch.tensor, size: int, sigma: float) -> torch.tensor:
        kernel = self.__gauss_kernel(size, sigma).cuda()
        kernel = kernel.view(1, 1, size, size)
        kernel = kernel.repeat(pert.shape[1], 1, 1, 1)
        blurred_pert = F.conv2d(pert, kernel, padding=size // 2, groups=pert.shape[1])

        return blurred_pert.squeeze(0)

    def __jpeg_compress(self, pert: torch.tensor, ratio: int) -> torch.tensor:
        pert_np = pert.detach().cpu().numpy().transpose(0, 2, 3, 1)
        pert_np = np.clip(pert_np * 255.0, 0, 255).astype("uint8")
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(ratio)]
        for i in range(pert_np.shape[0]):
            _, encimg = cv2.imencode(".jpg", pert_np[i], encode_params)
            pert_np[i] = cv2.imdecode(encimg, 1)

        compress_pert = pert_np.transpose((0, 3, 1, 2))
        compress_pert = torch.from_numpy(compress_pert / 255.0).float().cuda()

        return compress_pert

    def __rotate(self, pert: torch.tensor, angle: float) -> None:
        import torchvision.transforms.functional as F

        rotated_tensor = torch.stack([F.rotate(tensor, angle) for tensor in pert])

        return rotated_tensor

    def __calculate_robustness_effectiveness(
        self,
        source_imgs: torch.tensor,
        pert_robust_imgs_swap: torch.tensor,
    ):
        source_imgs_ndarray = (
            source_imgs.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255.0
        )
        pert_robust_imgs_ndarray = (
            pert_robust_imgs_swap.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255.0
        )

        effectiveness, _ = self.effectiveness.compare(
            source_imgs_ndarray, pert_robust_imgs_ndarray
        )

        return effectiveness

    def __save_robustness_samples(
        self, experiment: str, imgs: list[torch.tensor]
    ) -> None:
        img_names = [
            "source",
            "target",
            "swap",
            "pert",
            "pert_swap",
            experiment,
            f"{experiment}_swap",
        ]

        for i, name in enumerate(img_names):
            for j in range(imgs[i].shape[0]):
                save_image(
                    imgs[i][j], join(self.args.log_dir, "image", f"{name}_{j}.png")
                )

        results = torch.cat(imgs, dim=0)
        save_image(
            results,
            join(self.args.log_dir, "image", f"{experiment}_summary.png"),
            nrow=imgs[0].shape[0],
        )
        del results

    def pgd_source_robustness_sample(self, loss_weights=[1, 1]):
        self.logger.info(f"loss_weights: {loss_weights}")

        gauss_mean, gauss_std = 0, 0.1
        gauss_size, gauss_sigma = 5, 3.0
        jpeg_ratio = 70
        rotate_angle = 60

        self.target.cuda().eval()
        l2_loss = nn.MSELoss().cuda()

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

        x_imgs = source_imgs.clone().detach()
        epsilon = (
            self.args.pgd_epsilon
            * (torch.max(source_imgs) - torch.min(source_imgs))
            / 2
        )

        mimic_img = self._load_imgs([join(self.args.data_dir, self.args.pgd_mimic)])
        mimic_img_expand = mimic_img.repeat(len(source_path), 1, 1, 1)
        mimic_identity_expand = self._get_imgs_identity(mimic_img_expand)
        for epoch in range(self.args.epochs):
            x_imgs.requires_grad = True

            x_identity = torch.empty(len(source_path), 512).cuda()
            for j in range(len(source_path)):
                identity = self._get_imgs_identity(x_imgs[j].unsqueeze(0))
                x_identity[j] = identity[0]
                del identity

            pert_diff_loss = l2_loss(x_imgs, source_imgs.detach())
            identity_diff_loss = l2_loss(x_identity, mimic_identity_expand.detach())

            loss = (
                loss_weights[0] * pert_diff_loss + loss_weights[1] * identity_diff_loss
            )
            loss.backward()

            x_imgs = (
                x_imgs.clone().detach() - epsilon * x_imgs.grad.sign().clone().detach()
            )
            x_imgs = torch.clamp(
                x_imgs,
                min=source_imgs - self.args.pgd_limit,
                max=source_imgs + self.args.pgd_limit,
            )

            self.logger.info(
                f"[Epoch {epoch+1:3}/{self.args.epochs:3}]loss: {loss:.5f}({loss_weights[0] * pert_diff_loss.item():.5f}, {loss_weights[1] * identity_diff_loss.item():.5f})"
            )

        x_identity = self._get_imgs_identity(x_imgs)
        x_swap_img = self.target(None, target_imgs, x_identity, None, True)

        noise_imgs = self.__gauss_noise(x_imgs, gauss_mean, gauss_std)
        noise_identity = self._get_imgs_identity(noise_imgs)
        noise_swap_imgs = self.target(None, target_imgs, noise_identity, None, True)
        noise_mse, noise_psnr, noise_ssim = self._calculate_utility(
            swap_imgs, noise_swap_imgs
        )
        noise_effec = self.__calculate_robustness_effectiveness(
            source_imgs, noise_swap_imgs
        )
        self.__save_robustness_samples(
            "noise",
            [
                source_imgs,
                target_imgs,
                swap_imgs,
                x_imgs,
                x_swap_img,
                noise_imgs,
                noise_swap_imgs,
            ],
        )
        del noise_imgs, noise_identity, noise_swap_imgs

        blur_imgs = self.__gauss_blur(x_imgs, gauss_size, gauss_sigma)
        blur_identity = self._get_imgs_identity(blur_imgs)
        blur_swap_imgs = self.target(None, target_imgs, blur_identity, None, True)
        blur_mse, blur_psnr, blur_ssim = self._calculate_utility(
            swap_imgs, blur_swap_imgs
        )
        blur_effec = self.__calculate_robustness_effectiveness(
            source_imgs, blur_swap_imgs
        )
        self.__save_robustness_samples(
            "blur",
            [
                source_imgs,
                target_imgs,
                swap_imgs,
                x_imgs,
                x_swap_img,
                blur_imgs,
                blur_swap_imgs,
            ],
        )
        del blur_imgs, blur_identity, blur_swap_imgs

        compress_imgs = self.__jpeg_compress(x_imgs, jpeg_ratio)
        compress_identity = self._get_imgs_identity(compress_imgs)
        compress_swap_imgs = self.target(
            None, target_imgs, compress_identity, None, True
        )
        compress_mse, compress_psnr, compress_ssim = self._calculate_utility(
            swap_imgs, compress_swap_imgs
        )
        compress_effec = self.__calculate_robustness_effectiveness(
            source_imgs, compress_swap_imgs
        )
        self.__save_robustness_samples(
            "compress",
            [
                source_imgs,
                target_imgs,
                swap_imgs,
                x_imgs,
                x_swap_img,
                compress_imgs,
                compress_swap_imgs,
            ],
        )
        del compress_imgs, compress_identity, compress_swap_imgs

        rotate_imgs = self.__rotate(x_imgs, rotate_angle)
        rotate_identity = self._get_imgs_identity(rotate_imgs)
        rotate_swap_imgs = self.target(None, target_imgs, rotate_identity, None, True)
        rotate_mse, rotate_psnr, rotate_ssim = self._calculate_utility(
            swap_imgs, rotate_swap_imgs
        )
        rotate_effec = self.__calculate_robustness_effectiveness(
            source_imgs, rotate_swap_imgs
        )
        self.__save_robustness_samples(
            "rotate",
            [
                source_imgs,
                target_imgs,
                swap_imgs,
                x_imgs,
                x_swap_img,
                rotate_imgs,
                rotate_swap_imgs,
            ],
        )
        del rotate_imgs, rotate_identity, rotate_swap_imgs

        torch.cuda.empty_cache()
        self.logger.info(
            f"noise, blur, compress, rotate(mse, psnr, ssim, effectiveness): ({noise_mse:.3f}, {noise_psnr:.3f}, {noise_ssim:.3f}, {noise_effec:.3f}), ({blur_mse:.3f}, {blur_psnr:.3f}, {blur_ssim:.3f}, {blur_effec:.3f}), ({compress_mse:.3f}, {compress_psnr:.3f}, {compress_ssim:.3f}, {compress_effec:.3f}), ({rotate_mse:.3f}, {rotate_psnr:.3f}, {rotate_ssim:.3f}, {rotate_effec:.3f})"
        )

    def pgd_source_robustness_metric(self, loss_weights=[1, 1]):
        self.logger.info(f"loss_weights: {loss_weights}")

        gauss_mean, gauss_std = 0, 0.1
        gauss_size, gauss_sigma = 5, 3.0
        jpeg_ratio = 70
        rotate_angle = 60

        self.target.cuda().eval()
        l2_loss = nn.MSELoss().cuda()

        source_imgs_path, target_imgs_path = self._get_split_test_imgs_path()
        utility = {  # pert swap (mse, psnr, ssim)
            "noise": (0, 0, 0),
            "blur": (0, 0, 0),
            "compress": (0, 0, 0),
            "rotate": (0, 0, 0),
        }
        effectiveness = {  # pert swap
            "noise": 0,
            "blur": 0,
            "compress": 0,
            "rotate": 0,
        }

        total_batch = (
            min(len(source_imgs_path), len(target_imgs_path)) // self.args.batch_size
        )

        mimic_img = self._load_imgs([join(self.args.data_dir, self.args.pgd_mimic)])
        mimic_img_expand = mimic_img.repeat(self.args.batch_size, 1, 1, 1)
        mimic_identity_expand = self._get_imgs_identity(mimic_img_expand)
        for i in range(total_batch):
            iter_source_path = source_imgs_path[
                i * self.args.batch_size : (i + 1) * self.args.batch_size
            ]
            iter_target_path = target_imgs_path[
                i * self.args.batch_size : (i + 1) * self.args.batch_size
            ]

            source_imgs = self._load_imgs(iter_source_path)
            target_imgs = self._load_imgs(iter_target_path)
            source_identity = self._get_imgs_identity(source_imgs)
            swap_imgs = self.target(None, target_imgs, source_identity, None, True)

            x_imgs = source_imgs.clone().detach()
            epsilon = (
                self.args.pgd_epsilon
                * (torch.max(source_imgs) - torch.min(source_imgs))
                / 2
            )

            for epoch in range(self.args.epochs):
                x_imgs.requires_grad = True

                x_identity = torch.empty(self.args.batch_size, 512).cuda()
                for j in range(self.args.batch_size):
                    identity = self._get_imgs_identity(x_imgs[j].unsqueeze(0))
                    x_identity[j] = identity[0]
                    del identity

                pert_diff_loss = l2_loss(x_imgs, source_imgs.detach())
                identity_diff_loss = l2_loss(x_identity, mimic_identity_expand.detach())

                loss = (
                    loss_weights[0] * pert_diff_loss
                    + loss_weights[1] * identity_diff_loss
                )
                loss.backward()

                x_imgs = (
                    x_imgs.clone().detach()
                    - epsilon * x_imgs.grad.sign().clone().detach()
                )
                x_imgs = torch.clamp(
                    x_imgs,
                    min=source_imgs - self.args.pgd_limit,
                    max=source_imgs + self.args.pgd_limit,
                )

                self.logger.info(
                    f"Iter {i:5}/{total_batch:5}[Epoch {epoch+1:3}/{self.args.epochs:3}]loss: {loss:.5f}({loss_weights[0] * pert_diff_loss.item():.5f}, {loss_weights[1] * identity_diff_loss.item():.5f})"
                )

            noise_imgs = self.__gauss_noise(x_imgs, gauss_mean, gauss_std)
            noise_identity = self._get_imgs_identity(noise_imgs)
            noise_swap_imgs = self.target(None, target_imgs, noise_identity, None, True)
            noise_mse, noise_psnr, noise_ssim = self._calculate_utility(
                swap_imgs, noise_swap_imgs
            )
            utility["noise"] = tuple(
                a + b
                for a, b in zip(
                    utility["noise"],
                    (
                        noise_mse,
                        noise_psnr,
                        noise_ssim,
                    ),
                )
            )
            noise_effec = self.__calculate_robustness_effectiveness(
                source_imgs, noise_swap_imgs
            )
            effectiveness["noise"] += noise_effec
            del noise_imgs, noise_identity, noise_swap_imgs

            blur_imgs = self.__gauss_blur(x_imgs, gauss_size, gauss_sigma)
            blur_identity = self._get_imgs_identity(blur_imgs)
            blur_swap_imgs = self.target(None, target_imgs, blur_identity, None, True)
            blur_mse, blur_psnr, blur_ssim = self._calculate_utility(
                swap_imgs, blur_swap_imgs
            )
            utility["blur"] = tuple(
                a + b
                for a, b in zip(
                    utility["blur"],
                    (
                        blur_mse,
                        blur_psnr,
                        blur_ssim,
                    ),
                )
            )
            blur_effec = self.__calculate_robustness_effectiveness(
                source_imgs, blur_swap_imgs
            )
            effectiveness["blur"] += blur_effec
            del blur_imgs, blur_identity, blur_swap_imgs

            compress_imgs = self.__jpeg_compress(x_imgs, jpeg_ratio)
            compress_identity = self._get_imgs_identity(compress_imgs)
            compress_swap_imgs = self.target(
                None, target_imgs, compress_identity, None, True
            )
            compress_mse, compress_psnr, compress_ssim = self._calculate_utility(
                swap_imgs, compress_swap_imgs
            )
            utility["compress"] = tuple(
                a + b
                for a, b in zip(
                    utility["compress"],
                    (
                        compress_mse,
                        compress_psnr,
                        compress_ssim,
                    ),
                )
            )
            compress_effec = self.__calculate_robustness_effectiveness(
                source_imgs, compress_swap_imgs
            )
            effectiveness["compress"] += compress_effec
            del compress_imgs, compress_identity, compress_swap_imgs

            rotate_imgs = self.__rotate(rotate_angle)
            rotate_identity = self._get_imgs_identity(rotate_imgs)
            rotate_swap_imgs = self.target(
                None, target_imgs, rotate_identity, None, True
            )
            rotate_mse, rotate_psnr, rotate_ssim = self._calculate_utility(
                swap_imgs, rotate_swap_imgs
            )
            utility["rotate"] = tuple(
                a + b
                for a, b in zip(
                    utility["rotate"],
                    (
                        rotate_mse,
                        rotate_psnr,
                        rotate_ssim,
                    ),
                )
            )
            rotate_effec = self.__calculate_robustness_effectiveness(
                source_imgs, rotate_swap_imgs
            )
            effectiveness["rotate"] += rotate_effec
            del rotate_imgs, rotate_identity, rotate_swap_imgs

            torch.cuda.empty_cache()
            self.logger.info(
                f"Iter {i:5}/{total_batch:5}, noise, blur, compress, rotate(mse, psnr, ssim, effectiveness): ({noise_mse:.3f}, {noise_psnr:.3f}, {noise_ssim:.3f}, {noise_effec:.3f}), ({blur_mse:.3f}, {blur_psnr:.3f}, {blur_ssim:.3f}, {blur_effec:.3f}), ({compress_mse:.3f}, {compress_psnr:.3f}, {compress_ssim:.3f}, {compress_effec:.3f}), ({rotate_mse:.3f}, {rotate_psnr:.3f}, {rotate_ssim:.3f}, {rotate_effec:.3f})"
            )

            self.logger.info(
                f"Average of {self.args.batch_size * (i + 1)} pictures, noise, blur, compress, rotate(mse, psnr, ssim, effectiveness): ({utility['noise'][0]/total_batch:.3f}, {utility['noise'][1]/total_batch:.3f}, {utility['noise'][2]/total_batch:.3f}, {effectiveness['noise']/total_batch:.3f}), ({utility['blur'][0]/total_batch:.3f}, {utility['blur'][1]/total_batch:.3f}, {utility['blur'][2]/total_batch:.3f}, {effectiveness['blur']/total_batch:.3f}), ({utility['compress'][0]/total_batch:.3f}, {utility['compress'][1]/total_batch:.3f}, {utility['compress'][2]/total_batch:.3f}, {effectiveness['compress']/total_batch:.3f}), ({utility['rotate'][0]/total_batch:.3f}, {utility['rotate'][1]/total_batch:.3f}, {utility['rotate'][2]/total_batch:.3f}, {effectiveness['rotate']/total_batch:.3f})"
            )

    def pgd_target_robustness_sample(self, loss_weights=[5000, 1]):
        self.logger.info(f"loss_weights: {loss_weights}")

        gauss_mean, gauss_std = 0, 0.1
        gauss_size, gauss_sigma = 5, 3.0
        jpeg_ratio = 70
        rotate_angle = 60

        self.target.cuda().eval()
        l2_loss = nn.MSELoss().cuda()

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

        x_imgs = target_imgs.clone().detach()
        epsilon = (
            self.args.pgd_epsilon
            * (torch.max(target_imgs) - torch.min(target_imgs))
            / 2
        )

        mimic_img = self._load_imgs([join(self.args.data_dir, self.args.pgd_mimic)])
        mimic_img_expand = mimic_img.repeat(len(source_path), 1, 1, 1)
        for epoch in range(self.args.epochs):
            x_imgs.requires_grad = True

            x_latent_code = self.target.netG.encoder(x_imgs)
            mimic_latent_code = self.target.netG.encoder(mimic_img_expand)
            pert_diff_loss = l2_loss(x_imgs, target_imgs.detach())
            latent_code_diff_loss = l2_loss(x_latent_code, mimic_latent_code.detach())
            loss = (
                loss_weights[0] * pert_diff_loss
                - loss_weights[1] * latent_code_diff_loss
            )

            loss.backward(retain_graph=True)

            x_imgs = (
                x_imgs.clone().detach() - epsilon * x_imgs.grad.sign().clone().detach()
            )
            x_imgs = torch.clamp(
                x_imgs,
                min=target_imgs - self.args.pgd_limit,
                max=target_imgs + self.args.pgd_limit,
            )
            self.logger.info(
                f"[Epoch {epoch:4}]loss: {loss:.5f}({loss_weights[0] * pert_diff_loss.item():.5f}, {loss_weights[1] * latent_code_diff_loss.item():.5f})"
            )

        x_swap_img = self.target(None, x_imgs, source_identity, None, True)

        noise_imgs = self.__gauss_noise(x_imgs, gauss_mean, gauss_std)
        noise_swap_imgs = self.target(None, noise_imgs, source_identity, None, True)
        noise_mse, noise_psnr, noise_ssim = self._calculate_utility(
            swap_imgs, noise_swap_imgs
        )
        noise_effec = self.__calculate_robustness_effectiveness(
            source_imgs, noise_swap_imgs
        )
        self.__save_robustness_samples(
            "noise",
            [
                source_imgs,
                target_imgs,
                swap_imgs,
                x_imgs,
                x_swap_img,
                noise_imgs,
                noise_swap_imgs,
            ],
        )
        del noise_imgs, noise_swap_imgs

        blur_imgs = self.__gauss_blur(x_imgs, gauss_size, gauss_sigma)
        blur_swap_imgs = self.target(None, blur_imgs, source_identity, None, True)
        blur_mse, blur_psnr, blur_ssim = self._calculate_utility(
            swap_imgs, blur_swap_imgs
        )
        blur_effec = self.__calculate_robustness_effectiveness(
            source_imgs, blur_swap_imgs
        )
        self.__save_robustness_samples(
            "blur",
            [
                source_imgs,
                target_imgs,
                swap_imgs,
                x_imgs,
                x_swap_img,
                blur_imgs,
                blur_swap_imgs,
            ],
        )
        del blur_imgs, blur_swap_imgs

        compress_imgs = self.__jpeg_compress(x_imgs, jpeg_ratio)
        compress_swap_imgs = self.target(
            None, compress_imgs, source_identity, None, True
        )
        compress_mse, compress_psnr, compress_ssim = self._calculate_utility(
            swap_imgs, compress_swap_imgs
        )
        compress_effec = self.__calculate_robustness_effectiveness(
            source_imgs, compress_swap_imgs
        )
        self.__save_robustness_samples(
            "compress",
            [
                source_imgs,
                target_imgs,
                swap_imgs,
                x_imgs,
                x_swap_img,
                compress_imgs,
                compress_swap_imgs,
            ],
        )
        del compress_imgs, compress_swap_imgs

        rotate_imgs = self.__rotate(x_imgs, rotate_angle)
        rotate_swap_imgs = self.target(None, rotate_imgs, source_identity, None, True)
        rotate_mse, rotate_psnr, rotate_ssim = self._calculate_utility(
            swap_imgs, rotate_swap_imgs
        )
        rotate_effec = self.__calculate_robustness_effectiveness(
            source_imgs, rotate_swap_imgs
        )
        self.__save_robustness_samples(
            "rotate",
            [
                source_imgs,
                target_imgs,
                swap_imgs,
                x_imgs,
                x_swap_img,
                rotate_imgs,
                rotate_swap_imgs,
            ],
        )
        del rotate_imgs, rotate_swap_imgs

        torch.cuda.empty_cache()
        self.logger.info(
            f"noise, blur, compress, rotate(mse, psnr, ssim, effectiveness): ({noise_mse:.3f}, {noise_psnr:.3f}, {noise_ssim:.3f}, {noise_effec:.3f}), ({blur_mse:.3f}, {blur_psnr:.3f}, {blur_ssim:.3f}, {blur_effec:.3f}), ({compress_mse:.3f}, {compress_psnr:.3f}, {compress_ssim:.3f}, {compress_effec:.3f}), ({rotate_mse:.3f}, {rotate_psnr:.3f}, {rotate_ssim:.3f}, {rotate_effec:.3f})"
        )

    def pgd_target_robustness_metric(self, loss_weights=[5000, 1]):
        self.logger.info(f"loss_weights: {loss_weights}")

        gauss_mean, gauss_std = 0, 0.1
        gauss_size, gauss_sigma = 5, 3.0
        jpeg_ratio = 70
        rotate_angle = 60

        self.target.cuda().eval()
        l2_loss = nn.MSELoss().cuda()

        source_imgs_path, target_imgs_path = self._get_split_test_imgs_path()
        utility = {  # pert swap (mse, psnr, ssim)
            "noise": (0, 0, 0),
            "blur": (0, 0, 0),
            "compress": (0, 0, 0),
            "rotate": (0, 0, 0),
        }
        effectiveness = {  # pert swap
            "noise": 0,
            "blur": 0,
            "compress": 0,
            "rotate": 0,
        }

        total_batch = (
            min(len(source_imgs_path), len(target_imgs_path)) // self.args.batch_size
        )
        mimic_img = self._load_imgs([join(self.args.data_dir, self.args.pgd_mimic)])
        mimic_img_expand = mimic_img.repeat(self.args.batch_size, 1, 1, 1)
        mimic_latent_code = self.target.netG.encoder(mimic_img_expand)
        for i in range(total_batch):
            iter_source_path = source_imgs_path[
                i * self.args.batch_size : (i + 1) * self.args.batch_size
            ]
            iter_target_path = target_imgs_path[
                i * self.args.batch_size : (i + 1) * self.args.batch_size
            ]

            source_imgs = self._load_imgs(iter_source_path)
            target_imgs = self._load_imgs(iter_target_path)
            source_identity = self._get_imgs_identity(source_imgs)
            swap_imgs = self.target(None, target_imgs, source_identity, None, True)

            x_imgs = target_imgs.clone().detach()
            epsilon = (
                self.args.pgd_epsilon
                * (torch.max(target_imgs) - torch.min(target_imgs))
                / 2
            )
            for epoch in range(self.args.epochs):
                x_imgs.requires_grad = True

                x_latent_code = self.target.netG.encoder(x_imgs)
                pert_diff_loss = l2_loss(x_imgs, target_imgs.detach())
                latent_code_diff_loss = l2_loss(
                    x_latent_code, mimic_latent_code.detach()
                )
                loss = (
                    loss_weights[0] * pert_diff_loss
                    - loss_weights[1] * latent_code_diff_loss
                )

                loss.backward(retain_graph=True)

                x_imgs = (
                    x_imgs.clone().detach()
                    - epsilon * x_imgs.grad.sign().clone().detach()
                )
                x_imgs = torch.clamp(
                    x_imgs,
                    min=target_imgs - self.args.pgd_limit,
                    max=target_imgs + self.args.pgd_limit,
                )

                self.logger.info(
                    f"Iter {i:5}/{total_batch:5}[Epoch {epoch+1:3}/{self.args.epochs:3}]loss: {loss:.5f}({loss_weights[0] * pert_diff_loss.item():.5f}, {loss_weights[1] * latent_code_diff_loss.item():.5f})"
                )

            x_swap_img = self.target(None, x_imgs, source_identity, None, True)

            noise_imgs = self.__gauss_noise(x_imgs, gauss_mean, gauss_std)
            noise_swap_imgs = self.target(None, noise_imgs, source_identity, None, True)
            noise_mse, noise_psnr, noise_ssim = self._calculate_utility(
                swap_imgs, noise_swap_imgs
            )
            noise_effec = self.__calculate_robustness_effectiveness(
                source_imgs, noise_swap_imgs
            )
            utility["noise"] = tuple(
                a + b
                for a, b in zip(
                    utility["noise"],
                    (
                        noise_mse,
                        noise_psnr,
                        noise_ssim,
                    ),
                )
            )
            effectiveness["noise"] += noise_effec
            del noise_imgs, noise_swap_imgs

            blur_imgs = self.__gauss_blur(x_imgs, gauss_size, gauss_sigma)
            blur_swap_imgs = self.target(None, blur_imgs, source_identity, None, True)
            blur_mse, blur_psnr, blur_ssim = self._calculate_utility(
                swap_imgs, blur_swap_imgs
            )
            blur_effec = self.__calculate_robustness_effectiveness(
                source_imgs, blur_swap_imgs
            )
            utility["blur"] = tuple(
                a + b
                for a, b in zip(
                    utility["blur"],
                    (
                        blur_mse,
                        blur_psnr,
                        blur_ssim,
                    ),
                )
            )
            effectiveness["blur"] += blur_effec
            del blur_imgs, blur_swap_imgs

            compress_imgs = self.__jpeg_compress(x_imgs, jpeg_ratio)
            compress_swap_imgs = self.target(
                None, compress_imgs, source_identity, None, True
            )
            compress_mse, compress_psnr, compress_ssim = self._calculate_utility(
                swap_imgs, compress_swap_imgs
            )
            compress_effec = self.__calculate_robustness_effectiveness(
                source_imgs, compress_swap_imgs
            )
            utility["compress"] = tuple(
                a + b
                for a, b in zip(
                    utility["compress"],
                    (
                        compress_mse,
                        compress_psnr,
                        compress_ssim,
                    ),
                )
            )
            effectiveness["compress"] += compress_effec
            del compress_imgs, compress_swap_imgs

            rotate_imgs = self.__rotate(x_imgs, rotate_angle)
            rotate_swap_imgs = self.target(
                None, rotate_imgs, source_identity, None, True
            )
            rotate_mse, rotate_psnr, rotate_ssim = self._calculate_utility(
                swap_imgs, rotate_swap_imgs
            )
            rotate_effec = self.__calculate_robustness_effectiveness(
                source_imgs, rotate_swap_imgs
            )
            utility["rotate"] = tuple(
                a + b
                for a, b in zip(
                    utility["rotate"],
                    (
                        rotate_mse,
                        rotate_psnr,
                        rotate_ssim,
                    ),
                )
            )
            effectiveness["rotate"] += rotate_effec
            del rotate_imgs, rotate_swap_imgs

            torch.cuda.empty_cache()
            self.logger.info(
                f"noise, blur, compress, rotate(mse, psnr, ssim, effectiveness): ({noise_mse:.3f}, {noise_psnr:.3f}, {noise_ssim:.3f}, {noise_effec:.3f}), ({blur_mse:.3f}, {blur_psnr:.3f}, {blur_ssim:.3f}, {blur_effec:.3f}), ({compress_mse:.3f}, {compress_psnr:.3f}, {compress_ssim:.3f}, {compress_effec:.3f}), ({rotate_mse:.3f}, {rotate_psnr:.3f}, {rotate_ssim:.3f}, {rotate_effec:.3f})"
            )

            self.logger.info(
                f"Average of {self.args.batch_size * (i + 1)} pictures, noise, blur, compress, rotate(mse, psnr, ssim, effectiveness): ({utility['noise'][0]/total_batch:.3f}, {utility['noise'][1]/total_batch:.3f}, {utility['noise'][2]/total_batch:.3f}, {effectiveness['noise']/total_batch:.3f}), ({utility['blur'][0]/total_batch:.3f}, {utility['blur'][1]/total_batch:.3f}, {utility['blur'][2]/total_batch:.3f}, {effectiveness['blur']/total_batch:.3f}), ({utility['compress'][0]/total_batch:.3f}, {utility['compress'][1]/total_batch:.3f}, {utility['compress'][2]/total_batch:.3f}, {effectiveness['compress']/total_batch:.3f}), ({utility['rotate'][0]/total_batch:.3f}, {utility['rotate'][1]/total_batch:.3f}, {utility['rotate'][2]/total_batch:.3f}, {effectiveness['rotate']/total_batch:.3f})"
            )

    def gan_source_robustness_sample(self) -> None:
        gauss_mean, gauss_std = 0, 0.1
        gauss_size, gauss_sigma = 5, 3.0
        jpeg_ratio = 70
        rotate_angle = 60

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

        pert_source_imgs = self.GAN_G(source_imgs)
        pert_source_identity = self._get_imgs_identity(pert_source_imgs)
        pert_swap_imgs = self.target(
            None, target_imgs, pert_source_identity, None, True
        )

        noise_imgs = self.__gauss_noise(pert_source_imgs, gauss_mean, gauss_std)
        noise_identity = self._get_imgs_identity(noise_imgs)
        noise_swap_imgs = self.target(None, target_imgs, noise_identity, None, True)
        noise_mse, noise_psnr, noise_ssim = self._calculate_utility(
            swap_imgs, noise_swap_imgs
        )
        noise_effec = self.__calculate_robustness_effectiveness(
            source_imgs, noise_swap_imgs
        )
        self.__save_robustness_samples(
            "noise",
            [
                source_imgs,
                target_imgs,
                swap_imgs,
                pert_source_imgs,
                pert_swap_imgs,
                noise_imgs,
                noise_swap_imgs,
            ],
        )
        del noise_imgs, noise_identity, noise_swap_imgs

        blur_imgs = self.__gauss_blur(pert_source_imgs, gauss_size, gauss_sigma)
        blur_identity = self._get_imgs_identity(blur_imgs)
        blur_swap_imgs = self.target(None, target_imgs, blur_identity, None, True)
        blur_mse, blur_psnr, blur_ssim = self._calculate_utility(
            swap_imgs, blur_swap_imgs
        )
        blur_effec = self.__calculate_robustness_effectiveness(
            source_imgs, blur_swap_imgs
        )
        self.__save_robustness_samples(
            "blur",
            [
                source_imgs,
                target_imgs,
                swap_imgs,
                pert_source_imgs,
                pert_swap_imgs,
                blur_imgs,
                blur_swap_imgs,
            ],
        )
        del blur_imgs, blur_identity, blur_swap_imgs

        compress_imgs = self.__jpeg_compress(pert_source_imgs, jpeg_ratio)
        compress_identity = self._get_imgs_identity(compress_imgs)
        compress_swap_imgs = self.target(
            None, target_imgs, compress_identity, None, True
        )
        compress_mse, compress_psnr, compress_ssim = self._calculate_utility(
            swap_imgs, compress_swap_imgs
        )
        compress_effec = self.__calculate_robustness_effectiveness(
            source_imgs, compress_swap_imgs
        )
        self.__save_robustness_samples(
            "compress",
            [
                source_imgs,
                target_imgs,
                swap_imgs,
                pert_source_imgs,
                pert_swap_imgs,
                compress_imgs,
                compress_swap_imgs,
            ],
        )
        del compress_imgs, compress_identity, compress_swap_imgs

        rotate_imgs = self.__rotate(pert_source_imgs, rotate_angle)
        rotate_identity = self._get_imgs_identity(rotate_imgs)
        rotate_swap_imgs = self.target(None, target_imgs, rotate_identity, None, True)
        rotate_mse, rotate_psnr, rotate_ssim = self._calculate_utility(
            swap_imgs, rotate_swap_imgs
        )
        rotate_effec = self.__calculate_robustness_effectiveness(
            source_imgs, rotate_swap_imgs
        )
        self.__save_robustness_samples(
            "rotate",
            [
                source_imgs,
                target_imgs,
                swap_imgs,
                pert_source_imgs,
                pert_swap_imgs,
                rotate_imgs,
                rotate_swap_imgs,
            ],
        )
        del rotate_imgs, rotate_identity, rotate_swap_imgs

        torch.cuda.empty_cache()
        self.logger.info(
            f"noise, blur, compress, rotate(mse, psnr, ssim, effectiveness): ({noise_mse:.3f}, {noise_psnr:.3f}, {noise_ssim:.3f}, {noise_effec:.3f}), ({blur_mse:.3f}, {blur_psnr:.3f}, {blur_ssim:.3f}, {blur_effec:.3f}), ({compress_mse:.3f}, {compress_psnr:.3f}, {compress_ssim:.3f}, {compress_effec:.3f}), ({rotate_mse:.3f}, {rotate_psnr:.3f}, {rotate_ssim:.3f}, {rotate_effec:.3f})"
        )

    def gan_target_robustness_sample(self) -> None:
        gauss_mean, gauss_std = 0, 0.1
        gauss_size, gauss_sigma = 5, 3.0
        jpeg_ratio = 70
        rotate_angle = 60

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

        pert_target_imgs = self.GAN_G(target_imgs)
        pert_swap_imgs = self.target(
            None, pert_target_imgs, source_identity, None, True
        )

        noise_imgs = self.__gauss_noise(pert_target_imgs, gauss_mean, gauss_std)
        noise_swap_imgs = self.target(None, noise_imgs, source_identity, None, True)
        noise_mse, noise_psnr, noise_ssim = self._calculate_utility(
            swap_imgs, noise_swap_imgs
        )
        noise_effec = self.__calculate_robustness_effectiveness(
            source_imgs, noise_swap_imgs
        )
        self.__save_robustness_samples(
            "noise",
            [
                source_imgs,
                target_imgs,
                swap_imgs,
                pert_target_imgs,
                pert_swap_imgs,
                noise_imgs,
                noise_swap_imgs,
            ],
        )
        del noise_imgs, noise_swap_imgs

        blur_imgs = self.__gauss_blur(pert_target_imgs, gauss_size, gauss_sigma)
        blur_swap_imgs = self.target(None, blur_imgs, source_identity, None, True)
        blur_mse, blur_psnr, blur_ssim = self._calculate_utility(
            swap_imgs, blur_swap_imgs
        )
        blur_effec = self.__calculate_robustness_effectiveness(
            source_imgs, blur_swap_imgs
        )
        self.__save_robustness_samples(
            "blur",
            [
                source_imgs,
                target_imgs,
                swap_imgs,
                pert_target_imgs,
                pert_swap_imgs,
                blur_imgs,
                blur_swap_imgs,
            ],
        )
        del blur_imgs, blur_swap_imgs

        compress_imgs = self.__jpeg_compress(pert_target_imgs, jpeg_ratio)
        compress_swap_imgs = self.target(
            None, compress_imgs, source_identity, None, True
        )
        compress_mse, compress_psnr, compress_ssim = self._calculate_utility(
            swap_imgs, compress_swap_imgs
        )
        compress_effec = self.__calculate_robustness_effectiveness(
            source_imgs, compress_swap_imgs
        )
        self.__save_robustness_samples(
            "compress",
            [
                source_imgs,
                target_imgs,
                swap_imgs,
                pert_target_imgs,
                pert_swap_imgs,
                compress_imgs,
                compress_swap_imgs,
            ],
        )
        del compress_imgs, compress_swap_imgs

        rotate_imgs = self.__rotate(pert_target_imgs, rotate_angle)
        rotate_swap_imgs = self.target(None, rotate_imgs, source_identity, None, True)
        rotate_mse, rotate_psnr, rotate_ssim = self._calculate_utility(
            swap_imgs, rotate_swap_imgs
        )
        rotate_effec = self.__calculate_robustness_effectiveness(
            source_imgs, rotate_swap_imgs
        )
        self.__save_robustness_samples(
            "rotate",
            [
                source_imgs,
                target_imgs,
                swap_imgs,
                pert_target_imgs,
                pert_swap_imgs,
                rotate_imgs,
                rotate_swap_imgs,
            ],
        )
        del rotate_imgs, rotate_swap_imgs

        torch.cuda.empty_cache()
        self.logger.info(
            f"noise, blur, compress, rotate(mse, psnr, ssim, effectiveness): ({noise_mse:.3f}, {noise_psnr:.3f}, {noise_ssim:.3f}, {noise_effec:.3f}), ({blur_mse:.3f}, {blur_psnr:.3f}, {blur_ssim:.3f}, {blur_effec:.3f}), ({compress_mse:.3f}, {compress_psnr:.3f}, {compress_ssim:.3f}, {compress_effec:.3f}), ({rotate_mse:.3f}, {rotate_psnr:.3f}, {rotate_ssim:.3f}, {rotate_effec:.3f})"
        )

    def gan_source_robustness_metric(self) -> None:
        gauss_mean, gauss_std = 0, 0.1
        gauss_size, gauss_sigma = 5, 3.0
        jpeg_ratio = 70
        rotate_angle = 60

        model_path = join("checkpoints", self.args.gan_test_models)
        self.GAN_G.load_state_dict(torch.load(model_path)["GAN_G_state_dict"])

        self.target.cuda().eval()
        self.GAN_G.cuda().eval()

        source_imgs_path, target_imgs_path = self._get_split_test_imgs_path()
        utility = {  # pert swap (mse, psnr, ssim)
            "noise": (0, 0, 0),
            "blur": (0, 0, 0),
            "compress": (0, 0, 0),
            "rotate": (0, 0, 0),
        }
        effectiveness = {  # pert swap
            "noise": 0,
            "blur": 0,
            "compress": 0,
            "rotate": 0,
        }

        total_batch = (
            min(len(source_imgs_path), len(target_imgs_path)) // self.args.batch_size
        )
        for i in range(total_batch):
            iter_source_path = source_imgs_path[
                i * self.args.batch_size : (i + 1) * self.args.batch_size
            ]
            iter_target_path = target_imgs_path[
                i * self.args.batch_size : (i + 1) * self.args.batch_size
            ]

            source_imgs = self._load_imgs(iter_source_path)
            target_imgs = self._load_imgs(iter_target_path)
            source_identity = self._get_imgs_identity(source_imgs)
            swap_imgs = self.target(None, target_imgs, source_identity, None, True)

            pert_imgs = self.GAN_G(source_imgs)

            noise_imgs = self.__gauss_noise(pert_imgs, gauss_mean, gauss_std)
            noise_identity = self._get_imgs_identity(noise_imgs)
            noise_swap_imgs = self.target(None, target_imgs, noise_identity, None, True)
            noise_mse, noise_psnr, noise_ssim = self._calculate_utility(
                swap_imgs, noise_swap_imgs
            )
            utility["noise"] = tuple(
                a + b
                for a, b in zip(
                    utility["noise"],
                    (
                        noise_mse,
                        noise_psnr,
                        noise_ssim,
                    ),
                )
            )
            noise_effec = self.__calculate_robustness_effectiveness(
                source_imgs, noise_swap_imgs
            )
            effectiveness["noise"] += noise_effec
            del noise_imgs, noise_identity, noise_swap_imgs

            blur_imgs = self.__gauss_blur(pert_imgs, gauss_size, gauss_sigma)
            blur_identity = self._get_imgs_identity(blur_imgs)
            blur_swap_imgs = self.target(None, target_imgs, blur_identity, None, True)
            blur_mse, blur_psnr, blur_ssim = self._calculate_utility(
                swap_imgs, blur_swap_imgs
            )
            utility["blur"] = tuple(
                a + b
                for a, b in zip(
                    utility["blur"],
                    (
                        blur_mse,
                        blur_psnr,
                        blur_ssim,
                    ),
                )
            )
            blur_effec = self.__calculate_robustness_effectiveness(
                source_imgs, blur_swap_imgs
            )
            effectiveness["blur"] += blur_effec
            del blur_imgs, blur_identity, blur_swap_imgs

            compress_imgs = self.__jpeg_compress(pert_imgs, jpeg_ratio)
            compress_identity = self._get_imgs_identity(compress_imgs)
            compress_swap_imgs = self.target(
                None, target_imgs, compress_identity, None, True
            )
            compress_mse, compress_psnr, compress_ssim = self._calculate_utility(
                swap_imgs, compress_swap_imgs
            )
            utility["compress"] = tuple(
                a + b
                for a, b in zip(
                    utility["compress"],
                    (
                        compress_mse,
                        compress_psnr,
                        compress_ssim,
                    ),
                )
            )
            compress_effec = self.__calculate_robustness_effectiveness(
                source_imgs, compress_swap_imgs
            )
            effectiveness["compress"] += compress_effec
            del compress_imgs, compress_identity, compress_swap_imgs

            rotate_imgs = self.__rotate(pert_imgs, rotate_angle)
            rotate_identity = self._get_imgs_identity(rotate_imgs)
            rotate_swap_imgs = self.target(
                None, target_imgs, rotate_identity, None, True
            )
            rotate_mse, rotate_psnr, rotate_ssim = self._calculate_utility(
                swap_imgs, rotate_swap_imgs
            )
            utility["rotate"] = tuple(
                a + b
                for a, b in zip(
                    utility["rotate"],
                    (
                        rotate_mse,
                        rotate_psnr,
                        rotate_ssim,
                    ),
                )
            )
            rotate_effec = self.__calculate_robustness_effectiveness(
                source_imgs, rotate_swap_imgs
            )
            effectiveness["rotate"] += rotate_effec
            del rotate_imgs, rotate_identity, rotate_swap_imgs

            torch.cuda.empty_cache()
            self.logger.info(
                f"Iter {i:5}/{total_batch:5}, noise, blur, compress, rotate(mse, psnr, ssim, effectiveness): ({noise_mse:.3f}, {noise_psnr:.3f}, {noise_ssim:.3f}, {noise_effec:.3f}), ({blur_mse:.3f}, {blur_psnr:.3f}, {blur_ssim:.3f}, {blur_effec:.3f}), ({compress_mse:.3f}, {compress_psnr:.3f}, {compress_ssim:.3f}, {compress_effec:.3f}), ({rotate_mse:.3f}, {rotate_psnr:.3f}, {rotate_ssim:.3f}, {rotate_effec:.3f})"
            )

            self.logger.info(
                f"Average of {self.args.batch_size * (i + 1)} pictures, noise, blur, compress, rotate(mse, psnr, ssim, effectiveness): ({utility['noise'][0]/total_batch:.3f}, {utility['noise'][1]/total_batch:.3f}, {utility['noise'][2]/total_batch:.3f}, {effectiveness['noise']/total_batch:.3f}), ({utility['blur'][0]/total_batch:.3f}, {utility['blur'][1]/total_batch:.3f}, {utility['blur'][2]/total_batch:.3f}, {effectiveness['blur']/total_batch:.3f}), ({utility['compress'][0]/total_batch:.3f}, {utility['compress'][1]/total_batch:.3f}, {utility['compress'][2]/total_batch:.3f}, {effectiveness['compress']/total_batch:.3f}), ({utility['rotate'][0]/total_batch:.3f}, {utility['rotate'][1]/total_batch:.3f}, {utility['rotate'][2]/total_batch:.3f}, {effectiveness['rotate']/total_batch:.3f})"
            )

    def gan_target_robustness_metric(self) -> None:
        gauss_mean, gauss_std = 0, 0.1
        gauss_size, gauss_sigma = 5, 3.0
        jpeg_ratio = 70
        rotate_angle = 60

        model_path = join("checkpoints", self.args.gan_test_models)
        self.GAN_G.load_state_dict(torch.load(model_path)["GAN_G_state_dict"])

        self.target.cuda().eval()
        self.GAN_G.cuda().eval()

        source_imgs_path, target_imgs_path = self._get_split_test_imgs_path()
        utility = {  # pert swap (mse, psnr, ssim)
            "noise": (0, 0, 0),
            "blur": (0, 0, 0),
            "compress": (0, 0, 0),
            "rotate": (0, 0, 0),
        }
        effectiveness = {  # pert swap
            "noise": 0,
            "blur": 0,
            "compress": 0,
            "rotate": 0,
        }

        total_batch = (
            min(len(source_imgs_path), len(target_imgs_path)) // self.args.batch_size
        )
        for i in range(total_batch):
            iter_source_path = source_imgs_path[
                i * self.args.batch_size : (i + 1) * self.args.batch_size
            ]
            iter_target_path = target_imgs_path[
                i * self.args.batch_size : (i + 1) * self.args.batch_size
            ]

            source_imgs = self._load_imgs(iter_source_path)
            target_imgs = self._load_imgs(iter_target_path)
            source_identity = self._get_imgs_identity(source_imgs)
            swap_imgs = self.target(None, target_imgs, source_identity, None, True)

            pert_imgs = self.GAN_G(target_imgs)
            pert_swap_imgs = self.target(None, pert_imgs, source_identity, None, True)

            noise_imgs = self.__gauss_noise(pert_imgs, gauss_mean, gauss_std)
            noise_swap_imgs = self.target(None, noise_imgs, source_identity, None, True)
            noise_mse, noise_psnr, noise_ssim = self._calculate_utility(
                swap_imgs, noise_swap_imgs
            )
            utility["noise"] = tuple(
                a + b
                for a, b in zip(
                    utility["noise"],
                    (
                        noise_mse,
                        noise_psnr,
                        noise_ssim,
                    ),
                )
            )
            noise_effec = self.__calculate_robustness_effectiveness(
                source_imgs, noise_swap_imgs
            )
            effectiveness["noise"] += noise_effec

            blur_imgs = self.__gauss_blur(pert_imgs, gauss_size, gauss_sigma)
            blur_swap_imgs = self.target(None, blur_imgs, source_identity, None, True)
            blur_mse, blur_psnr, blur_ssim = self._calculate_utility(
                swap_imgs, blur_swap_imgs
            )
            utility["blur"] = tuple(
                a + b
                for a, b in zip(
                    utility["blur"],
                    (
                        blur_mse,
                        blur_psnr,
                        blur_ssim,
                    ),
                )
            )
            blur_effec = self.__calculate_robustness_effectiveness(
                source_imgs, blur_swap_imgs
            )
            effectiveness["blur"] += blur_effec

            compress_imgs = self.__jpeg_compress(pert_imgs, jpeg_ratio)
            compress_swap_imgs = self.target(
                None, compress_imgs, source_identity, None, True
            )
            compress_mse, compress_psnr, compress_ssim = self._calculate_utility(
                swap_imgs, compress_swap_imgs
            )
            utility["compress"] = tuple(
                a + b
                for a, b in zip(
                    utility["compress"],
                    (
                        compress_mse,
                        compress_psnr,
                        compress_ssim,
                    ),
                )
            )
            compress_effec = self.__calculate_robustness_effectiveness(
                source_imgs, compress_swap_imgs
            )
            effectiveness["compress"] += compress_effec

            rotate_imgs = self.__rotate(pert_imgs, rotate_angle)
            rotate_swap_imgs = self.target(
                None, rotate_imgs, source_identity, None, True
            )
            rotate_mse, rotate_psnr, rotate_ssim = self._calculate_utility(
                swap_imgs, rotate_swap_imgs
            )
            utility["rotate"] = tuple(
                a + b
                for a, b in zip(
                    utility["rotate"],
                    (
                        rotate_mse,
                        rotate_psnr,
                        rotate_ssim,
                    ),
                )
            )
            rotate_effec = self.__calculate_robustness_effectiveness(
                source_imgs, rotate_swap_imgs
            )
            effectiveness["rotate"] += rotate_effec

            torch.cuda.empty_cache()
            self.logger.info(
                f"Iter {i:5}/{total_batch:5}, compress, noise, rotate, blur(mse, psnr, ssim, effectiveness): ({compress_mse:.3f}, {compress_psnr:.3f}, {compress_ssim:.3f}, {compress_effec:.3f}), ({noise_mse:.3f}, {noise_psnr:.3f}, {noise_ssim:.3f}, {noise_effec:.3f}),  ({rotate_mse:.3f}, {rotate_psnr:.3f}, {rotate_ssim:.3f}, {rotate_effec:.3f}), ({blur_mse:.3f}, {blur_psnr:.3f}, {blur_ssim:.3f}, {blur_effec:.3f})"
            )

            self.logger.info(
                f"Average of {self.args.batch_size * (i + 1)} pictures, compress, noise, rotate, blur(mse, psnr, ssim, effectiveness): ({utility['compress'][0]/(i + 1):.3f}, {utility['compress'][1]/(i + 1):.3f}, {utility['compress'][2]/(i + 1):.3f}, {effectiveness['compress']/(i + 1):.3f}), ({utility['noise'][0]/(i + 1):.3f}, {utility['noise'][1]/(i + 1):.3f}, {utility['noise'][2]/(i + 1):.3f}, {effectiveness['noise']/(i + 1):.3f}), ({utility['rotate'][0]/(i + 1):.3f}, {utility['rotate'][1]/(i + 1):.3f}, {utility['rotate'][2]/(i + 1):.3f}, {effectiveness['rotate']/(i + 1):.3f}), ({utility['blur'][0]/(i + 1):.3f}, {utility['blur'][1]/(i + 1):.3f}, {utility['blur'][2]/(i + 1):.3f}, {effectiveness['blur']/(i + 1):.3f})"
            )

            if i % self.args.log_interval == 0:
                results = torch.cat(
                    (
                        source_imgs,
                        target_imgs,
                        swap_imgs,
                        pert_imgs,
                        pert_swap_imgs,
                        compress_imgs,
                        compress_swap_imgs,
                        noise_imgs,
                        noise_swap_imgs,
                        rotate_imgs,
                        rotate_swap_imgs,
                        blur_imgs,
                        blur_swap_imgs,
                    ),
                    dim=0,
                )
                save_image(
                    results,
                    join(self.args.log_dir, "image", f"summary_{i}.png"),
                    nrow=self.args.batch_size,
                )
                del results

            del source_imgs, target_imgs, swap_imgs, pert_imgs, pert_swap_imgs
            del compress_imgs, compress_swap_imgs
            del noise_imgs, noise_swap_imgs
            del rotate_imgs, rotate_swap_imgs
            del blur_imgs, blur_swap_imgs
