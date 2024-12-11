from common_base import Base

import os
import random
import cv2
import torch
import math

import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


from os.path import join
from torchvision.utils import save_image
from models.fs_networks import Generator
from torch import tensor


class SimSwapDefense(Base, nn.Module):
    def __init__(self, args, logger):
        super().__init__(args, logger)

        self.samples_dir = join(args.data_dir, "samples")
        self.dataset_dir = join(args.data_dir, "vggface2_crop_224")
        self.trainset_dir = join(args.data_dir, "train")
        self.testset_dir = join(args.data_dir, "test")
        self.anchorset_dir = join(args.data_dir, "anchor", args.anchor_dir)

        self.imgs1_path = [
            join(self.samples_dir, i) for i in ["zjl.jpg", "james.jpg", "source.png"]
        ]
        self.imgs2_path = [
            join(self.samples_dir, i) for i in ["6.jpg", "6.jpg", "6.jpg"]
        ]

        self.pgd_loss_weights = {"pert": 1000, "identity": 10000, "latent": 0.1}
        self.pgd_loss_limits = {"latent": 30}

        self.gan_rgb_limits = [0.075, 0.03, 0.075]
        self.gan_loss_weights = {
            "pert": 750,
            "identity": 375,
            "latent": 0.1,
        }
        self.gan_loss_limits = {"identity": 0.003, "latent": 30}

        self.GAN_G = Generator(input_nc=3, output_nc=3, epsilon=self.gan_rgb_limits)

    def __generate_iter_utility_log(self, utilities: dict) -> str:
        return f"""
        {tuple(f'{v:.5f}' for _,v in utilities.items())}
        """.strip()

    def __generate_summary_utility_log(self, data: dict, item: str, batch: int) -> str:
        return f"""
        {tuple(f'{x / (batch + 1):.5f}' for x in data[item])}
        """.strip()

    def __generate_iter_effectiveness_log(self, effectiveness: dict) -> str:
        content = ""
        for effec in effectiveness:
            content += f"{tuple(f'{v[0]/v[1]*100:.3f}/{v[1]}' for _,v in effectiveness[effec].items())} "

        return content

    def __generate_summary_effectiveness_log(self, data: dict, item: str) -> str:
        content = ""
        for effec in data[item]:
            content += f"{tuple(f'{v[0]/v[1]*100:.3f}/{v[1]}' for _,v in data[item][effec].items())} "

        return content

    def __generate_iter_robustness_log(self, source: dict, target: dict) -> str:
        print(source, target)
        quit()
        return f"""
        {tuple(f'{v[0]/v[1]*100:.3f}/{v[1]}' for _,v in source['facerec'].items())}, {tuple(f'{v[0]/v[1]*100:.3f}/{v[1]}' for _,v in source['face++'].items())}, {tuple(f'{v[0]/v[1]*100:.3f}/{v[1]}' for _,v in target['facerec'].items())}, {tuple(f'{v[0]/v[1]*100:.3f}/{v[1]}' for _,v in target['face++'].items())}
        """.strip()

    def __generate_summary_robustness_log(self, data: dict) -> str:
        source = data["pert_as_src_effectiveness"]
        target = data["pert_as_tgt_effectiveness"]
        return f"""
        {tuple(f'{v[0]/v[1]*100:.3f}/{v[1]}' for _,v in source['fafacerecce_recognition'].items())}, {tuple(f'{v[0]/v[1]*100:.3f}/{v[1]}' for _,v in source['face++'].items())}, {tuple(f'{v[0]/v[1]*100:.3f}/{v[1]}' for _,v in target['facerec'].items())}, {tuple(f'{v[0]/v[1]*100:.3f}/{v[1]}' for _,v in target['face++'].items())}
        """.strip()

    def __get_protect_both_swap_imgs(
        self, imgs1: tensor, imgs2: tensor, pert_imgs1: tensor
    ) -> tuple[tensor, tensor, tensor, tensor]:
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

    def __find_best_anchor(self, imgs: tensor, anchor_imgs: tensor) -> tensor:
        imgs_ndarray = imgs.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255.0
        anchor_img_ndarray = (
            anchor_imgs.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255.0
        )
        best_anchors = []
        for i in range(imgs.shape[0]):
            distances = []
            for j in range(anchor_imgs.shape[0]):
                distance = self.effectiveness.get_image_distance(
                    imgs_ndarray[i], anchor_img_ndarray[j]
                )
                if distance is math.nan:
                    continue
                distances.append((distance, j))

            sorted_distances = sorted(distances)
            if len(sorted_distances) > self.args.anchor_index:
                best_anchor_idx = sorted_distances[self.args.anchor_index][1]
                best_anchors.append(anchor_imgs[best_anchor_idx])
            else:
                best_anchors.append(anchor_imgs[0])

        return torch.stack(best_anchors, dim=0)

    def __perturb_pgd_imgs(
        self, imgs: tensor, best_anchor_imgs: tensor, silent: bool = False
    ) -> tuple[tensor, tensor]:
        l2_loss = nn.MSELoss().cuda()
        best_anchor_identity = self._get_imgs_identity(best_anchor_imgs)
        imgs_latent_code = self.target.netG.encoder(imgs)
        x_imgs = imgs.clone().detach()
        epsilon = self.args.pgd_epsilon * (torch.max(imgs) - torch.min(imgs)) / 2
        for epoch in range(self.args.epochs):
            x_imgs.requires_grad = True

            x_identity = self._get_imgs_identity(x_imgs)
            identity_diff_loss = l2_loss(x_identity, best_anchor_identity.detach())
            x_latent_code = self.target.netG.encoder(x_imgs)

            pert_diff_loss = l2_loss(x_imgs, imgs.clone().detach())
            identity_diff_loss = l2_loss(x_identity, best_anchor_identity.detach())
            latent_code_diff_loss = -torch.clamp(
                l2_loss(x_latent_code, imgs_latent_code.detach()),
                0.0,
                self.pgd_loss_limits["latent"],
            )

            loss = (
                self.pgd_loss_weights["pert"] * pert_diff_loss
                + self.pgd_loss_weights["identity"] * identity_diff_loss
                + self.pgd_loss_weights["latent"] * latent_code_diff_loss
            )
            loss.backward(retain_graph=True)

            x_imgs = (
                x_imgs.clone().detach() - epsilon * x_imgs.grad.sign().clone().detach()
            )
            x_imgs = torch.clamp(
                x_imgs,
                min=imgs - self.args.pgd_limit,
                max=imgs + self.args.pgd_limit,
            )

            if not silent:
                self.logger.info(
                    f"[Epoch {epoch+1:4}/{self.args.epochs:4}]loss: {loss:.5f}({self.pgd_loss_weights['pert'] * pert_diff_loss.item():.5f}, {self.pgd_loss_weights['identity'] * identity_diff_loss.item():.5f}, {self.pgd_loss_weights['latent'] * latent_code_diff_loss.item():.5f})"
                )

        return x_imgs

    def __save_pgd_summary(
        self,
        imgs1: tensor,
        imgs2: tensor,
        x_imgs: tensor,
        best_anchor_imgs: tensor,
        imgs1_src_swap: tensor,
        pert_imgs1_src_swap: tensor,
        imgs1_tgt_swap: tensor,
        pert_imgs1_tgt_swap: tensor,
    ) -> None:
        img_names = {
            "source": imgs1,
            "target": imgs2,
            "pert": x_imgs,
            "anchor": best_anchor_imgs,
            "swap": imgs1_src_swap,
            "pert_swap": pert_imgs1_src_swap,
            "reverse_swap": imgs1_tgt_swap,
            "pert_reverse_swap": pert_imgs1_tgt_swap,
        }

        for name, img in img_names.items():
            for i in range(img.shape[0]):
                save_image(img[i], join(self.args.log_dir, "image", f"{name}{i}.png"))

        results = torch.cat(
            (
                imgs1,
                imgs2,
                x_imgs,
                best_anchor_imgs,
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
            nrow=imgs1.shape[0],
        )
        del results

    def __calculate_pgd_metric(
        self,
        imgs1: tensor,
        imgs2: tensor,
        x_imgs: tensor,
        best_anchor_imgs: tensor,
        imgs1_src_swap: tensor,
        pert_imgs1_src_swap: tensor,
        imgs1_tgt_swap: tensor,
        pert_imgs1_tgt_swap: tensor,
    ) -> tuple[dict, dict, dict, dict, dict]:
        pert_utilities = self.utility.calculate_utility(imgs1, x_imgs)
        pert_as_src_swap_utilities = self.utility.calculate_utility(
            imgs1_src_swap, pert_imgs1_src_swap
        )
        pert_as_tgt_swap_utilities = self.utility.calculate_utility(
            imgs1_tgt_swap, pert_imgs1_tgt_swap
        )
        source_effectivenesses = self.effectiveness.calculate_effectiveness(
            imgs1,
            x_imgs,
            imgs1_src_swap,
            pert_imgs1_src_swap,
            best_anchor_imgs,
        )
        target_effectivenesses = self.effectiveness.calculate_effectiveness(
            imgs2, None, imgs1_tgt_swap, pert_imgs1_tgt_swap, None
        )

        return (
            pert_utilities,
            pert_as_src_swap_utilities,
            pert_as_tgt_swap_utilities,
            source_effectivenesses,
            target_effectivenesses,
        )

    def __save_robustness_samples(self, experiment: str, imgs: list[tensor]) -> None:
        img_names = [
            "img1",
            "img2",
            f"{experiment}",
            f"{experiment}_swap",
            f"reverse_{experiment}_swap",
            "pert",
            f"pert_{experiment}",
            f"{experiment}_pert_swap",
            f"reverse_{experiment}_pert_swap",
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

    def pgd_both_sample(self) -> None:
        self.logger.info(
            f"loss_weights: {self.pgd_loss_weights}, loss_limits: {self.pgd_loss_limits}"
        )

        self.target.cuda().eval()

        imgs1 = self._load_imgs(self.imgs1_path)
        best_anchor_imgs = self.anchor.find_best_anchors(imgs1)
        x_imgs = self.__perturb_pgd_imgs(imgs1, best_anchor_imgs)

        imgs2 = self._load_imgs(self.imgs2_path)
        imgs1_src_swap, pert_imgs1_src_swap, imgs1_tgt_swap, pert_imgs1_tgt_swap = (
            self.__get_protect_both_swap_imgs(imgs1, imgs2, x_imgs)
        )

        self.__save_pgd_summary(
            imgs1,
            imgs2,
            x_imgs,
            best_anchor_imgs,
            imgs1_src_swap,
            pert_imgs1_src_swap,
            imgs1_tgt_swap,
            pert_imgs1_tgt_swap,
        )

        (
            pert_utilities,
            pert_as_src_swap_utilities,
            pert_as_tgt_swap_utilities,
            source_effectivenesses,
            target_effectivenesses,
        ) = self.__calculate_pgd_metric(
            imgs1,
            imgs2,
            x_imgs,
            best_anchor_imgs,
            imgs1_src_swap,
            pert_imgs1_src_swap,
            imgs1_tgt_swap,
            pert_imgs1_tgt_swap,
        )

        self.logger.info(
            f"""
        utility(mse, psnr, ssim, lpips), effectiveness{self.effectiveness.candi_funcs.keys()} source(pert, swap, pert_swap, anchor) target(swap, pert_swap)
        pert utility: {self.__generate_iter_utility_log(pert_utilities)}
        pert as swap source utility: {self.__generate_iter_utility_log(pert_as_src_swap_utilities)}
        pert as swap target utility: {self.__generate_iter_utility_log(pert_as_tgt_swap_utilities)}
        pert as swap source effectiveness: {self.__generate_iter_effectiveness_log(source_effectivenesses)}
        pert as swap target effectiveness: {self.__generate_iter_effectiveness_log(target_effectivenesses)}
        """
        )

    def __merge_metric(
        self,
        data: dict,
        pert_utilities: dict,
        pert_as_src_swap_utilities: dict,
        pert_as_tgt_swap_utilities: dict,
        source_effectivenesses: dict,
        target_effectivenesses: dict,
    ) -> None:
        data["pert_utility"] = tuple(
            x + y
            for x, y in zip(
                data["pert_utility"],
                (
                    pert_utilities["mse"],
                    pert_utilities["psnr"],
                    pert_utilities["ssim"],
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
                    pert_as_src_swap_utilities["psnr"],
                    pert_as_src_swap_utilities["ssim"],
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
                    pert_as_tgt_swap_utilities["psnr"],
                    pert_as_tgt_swap_utilities["ssim"],
                    pert_as_tgt_swap_utilities["lpips"],
                ),
            )
        )

        for effec in self.effectiveness.candi_funcs.keys():
            data["pert_as_src_effectiveness"][effec] = {
                key1: (value1[0] + value2[0], value1[1] + value2[1])
                for (key1, value1), (key2, value2) in zip(
                    data["pert_as_src_effectiveness"][effec].items(),
                    source_effectivenesses[effec].items(),
                )
            }
            data["pert_as_tgt_effectiveness"][effec] = {
                key1: (value1[0] + value2[0], value1[1] + value2[1])
                for (key1, value1), (key2, value2) in zip(
                    data["pert_as_tgt_effectiveness"][effec].items(),
                    target_effectivenesses[effec].items(),
                )
            }

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

    def pgd_both_metric(
        self,
    ) -> None:
        self.logger.info(
            f"loss_weights: {self.pgd_loss_weights}, loss_limits: {self.pgd_loss_limits}"
        )

        self.target.cuda().eval()

        imgs1_path, imgs2_imgs_path = self._get_split_test_imgs_path()
        data = {
            "pert_utility": (0, 0, 0, 0),
            "pert_as_src_swap_utility": (0, 0, 0, 0),
            "pert_as_tgt_swap_utility": (0, 0, 0, 0),
            "pert_as_src_effectiveness": {},
            "pert_as_tgt_effectiveness": {},
        }

        for effec in self.effectiveness.candi_funcs.keys():
            data["pert_as_src_effectiveness"][effec] = {
                "pert": (0, 0),
                "swap": (0, 0),
                "pert_swap": (0, 0),
                "anchor": (0, 0),
            }
            data["pert_as_tgt_effectiveness"][effec] = {
                "swap": (0, 0),
                "pert_swap": (0, 0),
            }

        total_batch = min(len(imgs1_path), len(imgs2_imgs_path)) // self.args.batch_size
        for i in range(total_batch):
            iter_imgs1_path = imgs1_path[
                i * self.args.batch_size : (i + 1) * self.args.batch_size
            ]
            iter_imgs2_path = imgs2_imgs_path[
                i * self.args.batch_size : (i + 1) * self.args.batch_size
            ]

            imgs1 = self._load_imgs(iter_imgs1_path)
            imgs2 = self._load_imgs(iter_imgs2_path)

            best_anchor_imgs = self.anchor.find_best_anchors(imgs1)
            x_imgs = self.__perturb_pgd_imgs(imgs1, best_anchor_imgs, silent=True)

            imgs1_src_swap, pert_imgs1_src_swap, imgs1_tgt_swap, pert_imgs1_tgt_swap = (
                self.__get_protect_both_swap_imgs(imgs1, imgs2, x_imgs)
            )

            (
                pert_utilities,
                pert_as_src_swap_utilities,
                pert_as_tgt_swap_utilities,
                source_effectivenesses,
                target_effectivenesses,
            ) = self.__calculate_pgd_metric(
                imgs1,
                imgs2,
                x_imgs,
                best_anchor_imgs,
                imgs1_src_swap,
                pert_imgs1_src_swap,
                imgs1_tgt_swap,
                pert_imgs1_tgt_swap,
            )

            self.__merge_metric(
                data,
                pert_utilities,
                pert_as_src_swap_utilities,
                pert_as_tgt_swap_utilities,
                source_effectivenesses,
                target_effectivenesses,
            )

            del imgs1, imgs2, x_imgs, best_anchor_imgs
            del imgs1_src_swap, pert_imgs1_src_swap, imgs1_tgt_swap, pert_imgs1_tgt_swap
            torch.cuda.empty_cache()

            self.logger.info(
                f"""
            utility(mse, psnr, ssim, lpips), effectiveness{self.effectiveness.candi_funcs.keys()} source(pert, swap, pert_swap, anchor) target(swap, pert_swap)
            pert utility: {self.__generate_iter_utility_log(pert_utilities)}
            pert as swap source utility: {self.__generate_iter_utility_log(pert_as_src_swap_utilities)}
            pert as swap target utility: {self.__generate_iter_utility_log(pert_as_tgt_swap_utilities)}
            pert as swap source effectiveness: {self.__generate_iter_effectiveness_log(source_effectivenesses)}
            pert as swap target effectiveness: {self.__generate_iter_effectiveness_log(target_effectivenesses)}
            """
            )

            self.logger.info(
                f"""
            Batch {i + 1:4}/{total_batch:4}, {self.args.batch_size * (i + 1)} pairs of pictures
            {self.__generate_summary_utility_log(data, 'pert_utility', i)}
            {self.__generate_summary_utility_log(data, 'pert_as_src_swap_utility', i)}
            {self.__generate_summary_utility_log(data, 'pert_as_tgt_swap_utility', i)}
            {self.__generate_summary_effectiveness_log(data, 'pert_as_src_effectiveness')}
            {self.__generate_summary_effectiveness_log(data, 'pert_as_tgt_effectiveness')}
            """
            )

    def __gauss_noise(
        self, pert: tensor, gauss_mean: float, gauss_std: float
    ) -> tensor:
        gauss_noise = gauss_mean + gauss_std * torch.randn(pert.shape).cuda()
        noise_pert = pert + gauss_noise

        return noise_pert

    def __gauss_kernel(self, size: int, sigma: float):
        coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2.0
        grid = coords.repeat(size).view(size, size)
        kernel = torch.exp(-0.5 * (grid**2 + grid.t() ** 2) / sigma**2)
        kernel = kernel / kernel.sum()

        return kernel

    def __gauss_blur(self, pert: tensor, size: int, sigma: float) -> tensor:
        kernel = self.__gauss_kernel(size, sigma).cuda()
        kernel = kernel.view(1, 1, size, size)
        kernel = kernel.repeat(pert.shape[1], 1, 1, 1)
        blurred_pert = F.conv2d(pert, kernel, padding=size // 2, groups=pert.shape[1])

        return blurred_pert.squeeze(0)

    def __jpeg_compress(self, pert: tensor, ratio: int) -> tensor:
        pert_ndarray = pert.detach().cpu().numpy().transpose(0, 2, 3, 1)
        pert_ndarray = np.clip(pert_ndarray * 255.0, 0, 255).astype("uint8")
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(ratio)]
        for i in range(pert_ndarray.shape[0]):
            _, encimg = cv2.imencode(".jpg", pert_ndarray[i], encode_params)
            pert_ndarray[i] = cv2.imdecode(encimg, 1)

        compress_pert = pert_ndarray.transpose((0, 3, 1, 2))
        compress_pert = torch.from_numpy(compress_pert / 255.0).float().cuda()

        return compress_pert

    def __rotate(self, pert: tensor, angle: float) -> None:
        import torchvision.transforms.functional as F

        rotated_tensor = torch.stack([F.rotate(tensor, angle) for tensor in pert])

        return rotated_tensor

    def __get_gauss_noise_metrics(
        self,
        imgs1: tensor,
        imgs2: tensor,
        x_imgs: tensor,
        anchors_imgs: tensor,
    ) -> tuple[dict, dict, dict, dict]:
        gauss_mean, gauss_std = 0, 0.1

        noise_imgs = self.__gauss_noise(imgs1, gauss_mean, gauss_std)
        noise_identity = self._get_imgs_identity(noise_imgs)
        noise_pert_imgs = self.__gauss_noise(x_imgs, gauss_mean, gauss_std)
        noise_pert_identity = self._get_imgs_identity(noise_pert_imgs)

        imgs2_identity = self._get_imgs_identity(imgs2)
        noise_swap_imgs = self.target(None, imgs2, noise_identity, None, True)
        reverse_noise_swap_imgs = self.target(
            None, noise_imgs, imgs2_identity, None, True
        )
        noise_pert_swap_imgs = self.target(None, imgs2, noise_pert_identity, None, True)
        reverse_noise_pert_swap_imgs = self.target(
            None, noise_pert_imgs, imgs2_identity, None, True
        )

        source_effectivenesses = self.effectiveness.calculate_effectiveness(
            imgs1,
            x_imgs,
            noise_swap_imgs,
            noise_pert_swap_imgs,
            anchors_imgs,
        )
        target_effectivenesses = self.effectiveness.calculate_effectiveness(
            imgs2, None, reverse_noise_swap_imgs, reverse_noise_pert_swap_imgs, None
        )

        self.__save_robustness_samples(
            "noise",
            [
                imgs1,
                imgs2,
                noise_imgs,
                noise_swap_imgs,
                reverse_noise_swap_imgs,
                x_imgs,
                noise_pert_imgs,
                noise_pert_swap_imgs,
                reverse_noise_pert_swap_imgs,
            ],
        )

        return (
            source_effectivenesses,
            target_effectivenesses,
        )

    def __get_compress_metrics(
        self,
        imgs1: tensor,
        imgs2: tensor,
        x_imgs: tensor,
        anchors_imgs: tensor,
    ) -> tuple[dict, dict, dict, dict]:
        compress_rate = 85

        compress_imgs = self.__jpeg_compress(imgs1, compress_rate)
        compress_identity = self._get_imgs_identity(compress_imgs)
        compress_pert_imgs = self.__jpeg_compress(x_imgs, compress_rate)
        compress_pert_identity = self._get_imgs_identity(compress_pert_imgs)

        imgs2_identity = self._get_imgs_identity(imgs2)
        compress_swap_imgs = self.target(None, imgs2, compress_identity, None, True)
        reverse_compress_swap_imgs = self.target(
            None, compress_imgs, imgs2_identity, None, True
        )
        compress_pert_swap_imgs = self.target(
            None, imgs2, compress_pert_identity, None, True
        )
        reverse_compress_pert_swap_imgs = self.target(
            None, compress_pert_imgs, imgs2_identity, None, True
        )

        source_effectivenesses = self.effectiveness.calculate_effectiveness(
            imgs1,
            x_imgs,
            compress_swap_imgs,
            compress_pert_swap_imgs,
            anchors_imgs,
        )
        target_effectivenesses = self.effectiveness.calculate_effectiveness(
            imgs2,
            None,
            reverse_compress_swap_imgs,
            reverse_compress_pert_swap_imgs,
            None,
        )

        self.__save_robustness_samples(
            "compress",
            [
                imgs1,
                imgs2,
                compress_imgs,
                compress_swap_imgs,
                reverse_compress_swap_imgs,
                x_imgs,
                compress_pert_imgs,
                compress_pert_swap_imgs,
                reverse_compress_pert_swap_imgs,
            ],
        )

        return (
            source_effectivenesses,
            target_effectivenesses,
        )

    def __get_rotate_metrics(
        self,
        imgs1: tensor,
        imgs2: tensor,
        x_imgs: tensor,
        anchors_imgs: tensor,
    ) -> tuple[dict, dict, dict, dict]:
        rotate_angle = 60

        rotate_imgs = self.__rotate(imgs1, rotate_angle)
        rotate_identity = self._get_imgs_identity(rotate_imgs)
        rotate_pert_imgs = self.__rotate(x_imgs, rotate_angle)
        rotate_pert_identity = self._get_imgs_identity(rotate_pert_imgs)

        imgs2_identity = self._get_imgs_identity(imgs2)
        rotate_swap_imgs = self.target(None, imgs2, rotate_identity, None, True)
        reverse_rotate_swap_imgs = self.target(
            None, rotate_imgs, imgs2_identity, None, True
        )
        rotate_pert_swap_imgs = self.target(
            None, imgs2, rotate_pert_identity, None, True
        )
        reverse_rotate_pert_swap_imgs = self.target(
            None, rotate_pert_imgs, imgs2_identity, None, True
        )

        source_effectivenesses = self.effectiveness.calculate_effectiveness(
            imgs1,
            x_imgs,
            rotate_swap_imgs,
            rotate_pert_swap_imgs,
            anchors_imgs,
        )
        target_effectivenesses = self.effectiveness.calculate_effectiveness(
            imgs2, None, reverse_rotate_swap_imgs, reverse_rotate_pert_swap_imgs, None
        )

        self.__save_robustness_samples(
            "rotate",
            [
                imgs1,
                imgs2,
                rotate_imgs,
                rotate_swap_imgs,
                reverse_rotate_swap_imgs,
                x_imgs,
                rotate_pert_imgs,
                rotate_pert_swap_imgs,
                reverse_rotate_pert_swap_imgs,
            ],
        )

        return (
            source_effectivenesses,
            target_effectivenesses,
        )

    def pgd_both_robustness_sample(self):
        self.logger.info(
            f"loss_weights: {self.pgd_loss_weights}, loss_limits: {self.pgd_loss_limits}"
        )

        self.target.cuda().eval()

        imgs1 = self._load_imgs(self.imgs1_path)
        imgs2 = self._load_imgs(self.imgs2_path)

        best_anchor_imgs = self.anchor.find_best_anchors(imgs1)
        x_imgs = self.__perturb_pgd_imgs(imgs1, best_anchor_imgs)

        (
            noise_source_effectivenesses,
            noise_target_effectivenesses,
        ) = self.__get_gauss_noise_metrics(imgs1, imgs2, x_imgs, best_anchor_imgs)

        (
            compress_source_effectivenesses,
            compress_target_effectivenesses,
        ) = self.__get_compress_metrics(imgs1, imgs2, x_imgs, best_anchor_imgs)

        (
            rotate_source_effectivenesses,
            rotate_target_effectivenesses,
        ) = self.__get_rotate_metrics(imgs1, imgs2, x_imgs, best_anchor_imgs)

        torch.cuda.empty_cache()
        self.logger.info(
            f"""
            noise, compress, rotate effectiveness{self.effectiveness.candi_funcs.keys()}
            source(robust swap, robust pert swap, anchor), target(robust swap, robust pert swap)
            {self.__generate_iter_robustness_log(noise_source_effectivenesses,noise_target_effectivenesses)}
            {self.__generate_iter_robustness_log(compress_source_effectivenesses,compress_target_effectivenesses)}
            {self.__generate_iter_robustness_log(rotate_source_effectivenesses,rotate_target_effectivenesses)}
            """
        )

    def __merge_dict(self, sum: dict, item: dict):
        # sum and item must have identical structure
        for key in sum:
            if isinstance(sum[key], dict) and isinstance(item[key], dict):
                self.__merge_dict(sum[key], item[key])
            elif isinstance(sum[key], tuple) and isinstance(item[key], tuple):
                sum[key] = tuple(a + b for a, b in zip(sum[key], item[key]))
            else:
                sum[key] = sum[key] + item[key]

    def __merge_robustness_metric(
        self,
        data: dict,
        source_effectivenesses: dict,
        target_effectivenesses: dict,
        experiment: str,
    ) -> None:
        self.__merge_dict(
            data[experiment]["pert_as_src_effectiveness"], source_effectivenesses
        )
        self.__merge_dict(
            data[experiment]["pert_as_tgt_effectiveness"], target_effectivenesses
        )

    def pgd_both_robustness_metric(self):
        self.logger.info(
            f"loss_weights: {self.pgd_loss_weights}, loss_limits: {self.pgd_loss_limits}"
        )

        self.target.cuda().eval()

        imgs1_path, imgs2_imgs_path = self._get_split_test_imgs_path()
        robustness_data = {
            "pert_as_src_effectiveness": {
                "facerec": {
                    "swap": (0, 0),
                    "pert_swap": (0, 0),
                    "anchor": (0, 0),
                },
                "face++": {
                    "swap": (0, 0),
                    "pert_swap": (0, 0),
                    "anchor": (0, 0),
                },
            },
            "pert_as_tgt_effectiveness": {
                "facerec": {"swap": (0, 0), "pert_swap": (0, 0)},
                "face++": {"swap": (0, 0), "pert_swap": (0, 0)},
            },
        }
        from copy import deepcopy

        data = {
            "noise": deepcopy(robustness_data),
            "compress": deepcopy(robustness_data),
            "rotate": deepcopy(robustness_data),
        }

        anchor_imgs_path = self.__get_anchor_imgs_path()
        anchor_imgs = self._load_imgs(anchor_imgs_path)

        total_batch = min(len(imgs1_path), len(imgs2_imgs_path)) // self.args.batch_size
        for i in range(total_batch):
            iter_imgs1_path = imgs1_path[
                i * self.args.batch_size : (i + 1) * self.args.batch_size
            ]
            iter_imgs2_path = imgs2_imgs_path[
                i * self.args.batch_size : (i + 1) * self.args.batch_size
            ]

            imgs1 = self._load_imgs(iter_imgs1_path)
            imgs2 = self._load_imgs(iter_imgs2_path)

            best_anchor_imgs = self.anchor.find_best_anchors(imgs1)
            x_imgs = self.__perturb_pgd_imgs(imgs1, best_anchor_imgs, silent=True)

            (
                noise_source_effectivenesses,
                noise_target_effectivenesses,
            ) = self.__get_gauss_noise_metrics(imgs1, imgs2, x_imgs, best_anchor_imgs)

            self.__merge_robustness_metric(
                data,
                noise_source_effectivenesses,
                noise_target_effectivenesses,
                "noise",
            )

            (
                compress_source_effectivenesses,
                compress_target_effectivenesses,
            ) = self.__get_compress_metrics(imgs1, imgs2, x_imgs, best_anchor_imgs)
            self.__merge_robustness_metric(
                data,
                compress_source_effectivenesses,
                compress_target_effectivenesses,
                "compress",
            )

            (
                rotate_source_effectivenesses,
                rotate_target_effectivenesses,
            ) = self.__get_rotate_metrics(imgs1, imgs2, x_imgs, best_anchor_imgs)
            self.__merge_robustness_metric(
                data,
                rotate_source_effectivenesses,
                rotate_target_effectivenesses,
                "rotate",
            )

            torch.cuda.empty_cache()
            self.logger.info(
                f"""
            source face_rec(robust swap, robust pert swap, anchor), face++(robust swap, robust pert swap, anchor), target face_rec(swap, robust pert swap), face++(swap, robust pert swap)
            {self.__generate_iter_robustness_log(noise_source_effectivenesses,noise_target_effectivenesses)}
            {self.__generate_iter_robustness_log(compress_source_effectivenesses,compress_target_effectivenesses)}
            {self.__generate_iter_robustness_log(rotate_source_effectivenesses,rotate_target_effectivenesses)}
            """
            )

            self.logger.info(
                f"""[{i + 1}/{total_batch}]Average of {self.args.batch_size * (i + 1)} pictures(noise, compress, rotate)
            {self.__generate_summary_robustness_log(data['noise'])}
            {self.__generate_summary_robustness_log(data['compress'])}
            {self.__generate_summary_robustness_log(data['rotate'])}
            """
            )

    def __get_all_imgs_path(self, train_set: bool = True) -> list[str]:
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

    def gan_both_train(self):
        self.logger.info(
            f"rgb_limits: {self.gan_rgb_limits}, loss_weights: {self.gan_loss_weights}, loss_limits: {self.gan_loss_limits}"
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
        train_imgs_path = self.__get_all_imgs_path(train_set=True)
        test_imgs_path = self.__get_all_imgs_path(train_set=False)
        for epoch in range(self.args.epochs):
            self.GAN_G.cuda().train()
            imgs1_path = random.sample(train_imgs_path, self.args.batch_size)

            imgs1 = self._load_imgs(imgs1_path)
            imgs1_identity = self._get_imgs_identity(imgs1)
            imgs1_latent_code = self.target.netG.encoder(imgs1)

            pert_imgs1 = self.GAN_G(imgs1)
            pert_imgs1_identity = self._get_imgs_identity(pert_imgs1)
            pert_imgs1_latent_code = self.target.netG.encoder(pert_imgs1)

            self.GAN_G.zero_grad()

            pert_diff_loss = l2_loss(flatten(pert_imgs1), flatten(imgs1))
            identity_diff_loss = -torch.clamp(
                l2_loss(flatten(pert_imgs1_identity), flatten(imgs1_identity)),
                0.0,
                self.gan_loss_limits["identity"],
            )
            latent_diff_loss = -torch.clamp(
                l2_loss(flatten(pert_imgs1_latent_code), flatten(imgs1_latent_code)),
                0.0,
                self.gan_loss_limits["latent"],
            )

            G_loss = (
                self.gan_loss_weights["pert"] * pert_diff_loss
                + self.gan_loss_weights["identity"] * identity_diff_loss
                + self.gan_loss_weights["latent"] * latent_diff_loss
            )
            G_loss.backward(retain_graph=True)
            optimizer_G.step()
            scheduler.step()

            self.logger.info(
                f"[Epoch {epoch+1:6}]loss(pert, identity, latent, result): {G_loss:8.5f}({self.gan_loss_weights['pert'] * pert_diff_loss.item():.5f}, {self.gan_loss_weights['identity'] * identity_diff_loss.item():.5f}, {self.gan_loss_weights['latent'] * latent_diff_loss.item():.5f})({pert_diff_loss.item():.5f}, {identity_diff_loss.item():.5f}, {latent_diff_loss.item():.5f})"
            )

            if (epoch + 1) % self.args.log_interval == 0:
                with torch.no_grad():
                    self.GAN_G.eval()
                    self.target.eval()

                    test_imgs1_path = random.sample(test_imgs_path, 7)
                    test_imgs1_path.extend(self.imgs1_path)
                    test_imgs2_path = random.sample(test_imgs_path, 7)
                    test_imgs2_path.extend(self.imgs2_path)

                    test_imgs1 = self._load_imgs(test_imgs1_path)
                    test_imgs1_identity = self._get_imgs_identity(test_imgs1)
                    test_imgs2 = self._load_imgs(test_imgs2_path)
                    test_imgs2_identity = self._get_imgs_identity(test_imgs2)

                    test_swap_imgs = self.target(
                        None, test_imgs2, test_imgs1_identity, None, True
                    )
                    test_reverse_swap_imgs = self.target(
                        None, test_imgs1, test_imgs2_identity, None, True
                    )

                    test_pert_imgs1 = self.GAN_G(test_imgs1)
                    test_pert_imgs1_identity = self._get_imgs_identity(test_pert_imgs1)
                    test_pert_swap_imgs = self.target(
                        None, test_imgs2, test_pert_imgs1_identity, None, True
                    )
                    test_reverse_pert_swap_imgs = self.target(
                        None, test_pert_imgs1, test_imgs2_identity, None, True
                    )

                    results = torch.cat(
                        (
                            test_imgs1,
                            test_imgs2,
                            test_pert_imgs1,
                            test_swap_imgs,
                            test_pert_swap_imgs,
                            test_reverse_swap_imgs,
                            test_reverse_pert_swap_imgs,
                        ),
                        dim=0,
                    )

                    save_path = join(
                        self.args.log_dir, "image", f"gan_both_{epoch+1}.png"
                    )
                    save_image(results, save_path, nrow=10)

            if G_loss.data < best_loss:
                best_loss = G_loss.data
                log_save_path = join(self.args.log_dir, "checkpoint", "gan_both.pth")
                torch.save(
                    {
                        "epoch": epoch,
                        "GAN_G_state_dict": self.GAN_G.state_dict(),
                        "GAN_G_loss": G_loss,
                    },
                    log_save_path,
                )

    def __save_gan_summary(
        self,
        imgs1: tensor,
        imgs2: tensor,
        pert_imgs1: tensor,
        imgs1_src_swap: tensor,
        pert_imgs1_src_swap: tensor,
        imgs1_tgt_swap: tensor,
        pert_imgs1_tgt_swap: tensor,
    ) -> None:
        img_names = {
            "source": imgs1,
            "target": imgs2,
            "pert": pert_imgs1,
            "swap": imgs1_src_swap,
            "pert_swap": pert_imgs1_src_swap,
            "reverse_swap": imgs1_tgt_swap,
            "pert_reverse_swap": pert_imgs1_tgt_swap,
        }

        for name, img in img_names.items():
            for i in range(img.shape[0]):
                save_image(img[i], join(self.args.log_dir, "image", f"{name}{i}.png"))

        results = torch.cat(
            (
                imgs1,
                imgs2,
                pert_imgs1,
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
            nrow=imgs1.shape[0],
        )
        del results

    def __calculate_gan_metric(
        self,
        imgs1: tensor,
        imgs2: tensor,
        pert_imgs1: tensor,
        imgs1_src_swap: tensor,
        pert_imgs1_src_swap: tensor,
        imgs1_tgt_swap: tensor,
        pert_imgs1_tgt_swap: tensor,
    ) -> tuple[dict, dict, dict, dict, dict]:
        pert_utilities = self.utility.calculate_utility(imgs1, pert_imgs1)
        pert_as_src_swap_utilities = self.utility.calculate_utility(
            imgs1_src_swap, pert_imgs1_src_swap
        )
        pert_as_tgt_swap_utilities = self.utility.calculate_utility(
            imgs1_tgt_swap, pert_imgs1_tgt_swap
        )
        source_effectivenesses = self.effectiveness.calculate_effectiveness(
            imgs1,
            pert_imgs1,
            imgs1_src_swap,
            pert_imgs1_src_swap,
            None,
        )
        target_effectivenesses = self.effectiveness.calculate_effectiveness(
            imgs2,
            None,
            imgs1_tgt_swap,
            pert_imgs1_tgt_swap,
            None,
        )

        return (
            pert_utilities,
            pert_as_src_swap_utilities,
            pert_as_tgt_swap_utilities,
            source_effectivenesses,
            target_effectivenesses,
        )

    def gan_both_sample(self):
        model_path = join("checkpoints", self.args.gan_test_models)
        self.GAN_G.load_state_dict(torch.load(model_path)["GAN_G_state_dict"])

        self.target.cuda().eval()
        self.GAN_G.cuda().eval()

        imgs1 = self._load_imgs(self.imgs1_path)
        imgs2 = self._load_imgs(self.imgs2_path)
        pert_imgs1 = self.GAN_G(imgs1)

        imgs1_src_swap, pert_imgs1_src_swap, imgs1_tgt_swap, pert_imgs1_tgt_swap = (
            self.__get_protect_both_swap_imgs(imgs1, imgs2, pert_imgs1)
        )

        self.__save_gan_summary(
            imgs1,
            imgs2,
            pert_imgs1,
            imgs1_src_swap,
            pert_imgs1_src_swap,
            imgs1_tgt_swap,
            pert_imgs1_tgt_swap,
        )

        (
            pert_utilities,
            pert_as_src_swap_utilities,
            pert_as_tgt_swap_utilities,
            source_effectivenesses,
            target_effectivenesses,
        ) = self.__calculate_gan_metric(
            imgs1,
            imgs2,
            pert_imgs1,
            imgs1_src_swap,
            pert_imgs1_src_swap,
            imgs1_tgt_swap,
            pert_imgs1_tgt_swap,
        )

        adapt_effec_methods = [effec for effec in source_effectivenesses]
        self.logger.info(
            f"""
        utility(mse, psnr, ssim, lpips), effectiveness{adapt_effec_methods} source(pert, swap, pert_swap) target(swap, pert_swap)
        pert utility: {self.__generate_iter_utility_log(pert_utilities)}
        pert as swap source utility: {self.__generate_iter_utility_log(pert_as_src_swap_utilities)}
        pert as swap target utility: {self.__generate_iter_utility_log(pert_as_tgt_swap_utilities)}
        pert as swap source effectiveness: {self.__generate_iter_effectiveness_log(source_effectivenesses)}
        pert as swap target effectiveness: {self.__generate_iter_effectiveness_log(target_effectivenesses)}
        """
        )

    def gan_both_metric(self):
        model_path = join("checkpoints", self.args.gan_test_models)
        self.GAN_G.load_state_dict(torch.load(model_path)["GAN_G_state_dict"])

        self.target.cuda().eval()
        self.GAN_G.cuda().eval()

        imgs1_path, imgs2_imgs_path = self._get_split_test_imgs_path()
        data = {
            "pert_utility": (0, 0, 0, 0),
            "pert_as_src_swap_utility": (0, 0, 0, 0),
            "pert_as_tgt_swap_utility": (0, 0, 0, 0),
            "pert_as_src_effectiveness": {
                "facerec": {
                    "pert": (0, 0),
                    "swap": (0, 0),
                    "pert_swap": (0, 0),
                },
                "face++": {
                    "pert": (0, 0),
                    "swap": (0, 0),
                    "pert_swap": (0, 0),
                },
            },
            "pert_as_tgt_effectiveness": {
                "facerec": {"swap": (0, 0), "pert_swap": (0, 0)},
                "face++": {"swap": (0, 0), "pert_swap": (0, 0)},
            },
        }

        total_batch = min(len(imgs1_path), len(imgs2_imgs_path)) // self.args.batch_size
        for i in range(total_batch):
            iter_imgs1_path = imgs1_path[
                i * self.args.batch_size : (i + 1) * self.args.batch_size
            ]
            iter_imgs2_path = imgs2_imgs_path[
                i * self.args.batch_size : (i + 1) * self.args.batch_size
            ]

            imgs1 = self._load_imgs(iter_imgs1_path)
            imgs2 = self._load_imgs(iter_imgs2_path)

            pert_imgs1 = self.GAN_G(imgs1)
            imgs1_src_swap, pert_imgs1_src_swap, imgs1_tgt_swap, pert_imgs1_tgt_swap = (
                self.__get_protect_both_swap_imgs(imgs1, imgs2, pert_imgs1)
            )

            (
                pert_utilities,
                pert_as_src_swap_utilities,
                pert_as_tgt_swap_utilities,
                source_effectivenesses,
                target_effectivenesses,
            ) = self.__calculate_gan_metric(
                imgs1,
                imgs2,
                pert_imgs1,
                imgs1_src_swap,
                pert_imgs1_src_swap,
                imgs1_tgt_swap,
                pert_imgs1_tgt_swap,
            )

            self.__merge_metric(
                data,
                pert_utilities,
                pert_as_src_swap_utilities,
                pert_as_tgt_swap_utilities,
                source_effectivenesses,
                target_effectivenesses,
            )

            del imgs1, imgs2, pert_imgs1
            del imgs1_src_swap, pert_imgs1_src_swap, imgs1_tgt_swap, pert_imgs1_tgt_swap
            torch.cuda.empty_cache()

            adapt_effec_methods = [effec for effec in source_effectivenesses]
            self.logger.info(
                f"""
            utility(mse, psnr, ssim, lpips), effectiveness{adapt_effec_methods} source(pert, swap, pert_swap) target(swap, pert_swap)
            pert utility: {self.__generate_iter_utility_log(pert_utilities)}
            pert as swap source utility: {self.__generate_iter_utility_log(pert_as_src_swap_utilities)}
            pert as swap target utility: {self.__generate_iter_utility_log(pert_as_tgt_swap_utilities)}
            pert as swap source effectiveness: {self.__generate_iter_effectiveness_log(source_effectivenesses)}
            pert as swap target effectiveness: {self.__generate_iter_effectiveness_log(target_effectivenesses)}
            """
            )

            self.logger.info(
                f"""
            Batch {i + 1:4}/{total_batch:4}, {self.args.batch_size * (i + 1)} pairs of pictures
            {self.__generate_summary_utility_log(data, 'pert_utility', i)}
            {self.__generate_summary_utility_log(data, 'pert_as_src_swap_utility', i)}
            {self.__generate_summary_utility_log(data, 'pert_as_tgt_swap_utility', i)}
            {self.__generate_summary_effectiveness_log(data, 'pert_as_src_effectiveness')}
            {self.__generate_summary_effectiveness_log(data, 'pert_as_tgt_effectiveness')}
            """
            )

    def gan_both_robustness_sample(self):
        model_path = join("checkpoints", self.args.gan_test_models)
        self.GAN_G.load_state_dict(torch.load(model_path)["GAN_G_state_dict"])

        self.target.cuda().eval()
        self.GAN_G.cuda().eval()

        imgs1 = self._load_imgs(self.imgs1_path)
        imgs2 = self._load_imgs(self.imgs2_path)
        pert_imgs1 = self.GAN_G(imgs1)

        (
            noise_source_effectivenesses,
            noise_target_effectivenesses,
        ) = self.__get_gauss_noise_metrics(imgs1, imgs2, pert_imgs1, None)

        (
            compress_source_effectivenesses,
            compress_target_effectivenesses,
        ) = self.__get_compress_metrics(imgs1, imgs2, pert_imgs1, None)

        (
            rotate_source_effectivenesses,
            rotate_target_effectivenesses,
        ) = self.__get_rotate_metrics(imgs1, imgs2, pert_imgs1, None)

        torch.cuda.empty_cache()
        self.logger.info(
            f"""
            lines: noise, compress, rotate, tools: face recognition, face++
            source(robust swap, robust pert swap, anchor) target(robust swap, robust pert swap)
            {self.__generate_iter_robustness_log(noise_source_effectivenesses,noise_target_effectivenesses)}
            {self.__generate_iter_robustness_log(compress_source_effectivenesses,compress_target_effectivenesses)}
            {self.__generate_iter_robustness_log(rotate_source_effectivenesses,rotate_target_effectivenesses)}
            """
        )

    def gan_both_robustness_metric(self):
        model_path = join("checkpoints", self.args.gan_test_models)
        self.GAN_G.load_state_dict(torch.load(model_path)["GAN_G_state_dict"])

        self.target.cuda().eval()
        self.GAN_G.cuda().eval()

        imgs1_path, imgs2_imgs_path = self._get_split_test_imgs_path()
        robustness_data = {
            "pert_as_src_effectiveness": {
                "facerec": {
                    "swap": (0, 0),
                    "pert_swap": (0, 0),
                },
                "face++": {
                    "swap": (0, 0),
                    "pert_swap": (0, 0),
                },
            },
            "pert_as_tgt_effectiveness": {
                "facerec": {"swap": (0, 0), "pert_swap": (0, 0)},
                "face++": {"swap": (0, 0), "pert_swap": (0, 0)},
            },
        }
        from copy import deepcopy

        data = {
            "noise": deepcopy(robustness_data),
            "compress": deepcopy(robustness_data),
            "rotate": deepcopy(robustness_data),
        }

        total_batch = min(len(imgs1_path), len(imgs2_imgs_path)) // self.args.batch_size
        for i in range(total_batch):
            iter_imgs1_path = imgs1_path[
                i * self.args.batch_size : (i + 1) * self.args.batch_size
            ]
            iter_imgs2_path = imgs2_imgs_path[
                i * self.args.batch_size : (i + 1) * self.args.batch_size
            ]

            imgs1 = self._load_imgs(iter_imgs1_path)
            imgs2 = self._load_imgs(iter_imgs2_path)

            pert_imgs1 = self.GAN_G(imgs1)

            (
                noise_source_effectivenesses,
                noise_target_effectivenesses,
            ) = self.__get_gauss_noise_metrics(imgs1, imgs2, pert_imgs1, None)
            self.__merge_robustness_metric(
                data,
                noise_source_effectivenesses,
                noise_target_effectivenesses,
                "noise",
            )

            (
                compress_source_effectivenesses,
                compress_target_effectivenesses,
            ) = self.__get_compress_metrics(imgs1, imgs2, pert_imgs1, None)
            self.__merge_robustness_metric(
                data,
                compress_source_effectivenesses,
                compress_target_effectivenesses,
                "compress",
            )

            (
                rotate_source_effectivenesses,
                rotate_target_effectivenesses,
            ) = self.__get_rotate_metrics(imgs1, imgs2, pert_imgs1, None)
            self.__merge_robustness_metric(
                data,
                rotate_source_effectivenesses,
                rotate_target_effectivenesses,
                "rotate",
            )

            torch.cuda.empty_cache()
            self.logger.info(
                f"""
            lines: noise, compress, rotate, tools: face recognition, face++
            source(robust swap, robust pert swap, anchor) target(robust swap, robust pert swap)
            {self.__generate_iter_robustness_log(noise_source_effectivenesses,noise_target_effectivenesses)}
            {self.__generate_iter_robustness_log(compress_source_effectivenesses,compress_target_effectivenesses)}
            {self.__generate_iter_robustness_log(rotate_source_effectivenesses,rotate_target_effectivenesses)}
            """
            )

            self.logger.info(
                f"""[{i + 1}/{total_batch}]Average of {self.args.batch_size * (i + 1)} pictures(noise, compress, rotate)
            {self.__generate_summary_robustness_log(data['noise'])}
            {self.__generate_summary_robustness_log(data['compress'])}
            {self.__generate_summary_robustness_log(data['rotate'])}
            """
            )

    def __calculate_anchor_distance(
        self, imgs: tensor, anchor_imgs: tensor
    ) -> tuple[float, float]:
        imgs_ndarray = imgs.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255.0
        anchor_img_ndarray = (
            anchor_imgs.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255.0
        )

        count, min_dist, avg_dist = 0, 0, 0
        for i in range(imgs.shape[0]):
            distances = []
            for j in range(anchor_imgs.shape[0]):
                distance = self.effectiveness.get_image_distance(
                    imgs_ndarray[i], anchor_img_ndarray[j]
                )
                if distance is math.nan:
                    continue
                distances.append((distance, j))

            sorted_distances = sorted(distances)
            if len(sorted_distances) == 0:
                continue
            else:
                count += 1
                min_dist += sorted_distances[0][0]
                avg_dist += sorted_distances[int(len(sorted_distances) / 2)][0]

        return min_dist / count, avg_dist / count

    def calculate_distance(self):
        anchor_imgs_path = self.__get_anchor_imgs_path()
        anchor_imgs = self._load_imgs(anchor_imgs_path)
        min_distance, avg_distance = 0, 0

        imgs1_path, imgs2_imgs_path = self._get_split_test_imgs_path()
        total_batch = min(len(imgs1_path), len(imgs2_imgs_path)) // self.args.batch_size

        import tqdm

        for i in tqdm.tqdm(range(total_batch)):
            iter_imgs1_path = imgs1_path[
                i * self.args.batch_size : (i + 1) * self.args.batch_size
            ]
            imgs1 = self._load_imgs(iter_imgs1_path)
            distance = self.__calculate_anchor_distance(imgs1, anchor_imgs)
            if distance is not None:
                min_distance += distance[0]
                avg_distance += distance[1]
                tqdm.tqdm.write(f"min/avg distance {distance[0]}, {distance[1]}")

        self.logger.info(
            f"min and avg distance: {min_distance/total_batch:.5f}, {avg_distance/total_batch:.5f}"
        )

    def check_different_identity_effectiveness(self):
        anchor_imgs_path = self.__get_anchor_imgs_path()
        anchor_imgs = self._load_imgs(anchor_imgs_path)

        imgs1_path, imgs2_imgs_path = self._get_split_test_imgs_path()
        total_batch = min(len(imgs1_path), len(imgs2_imgs_path)) // self.args.batch_size

        import tqdm

        effectivenesses = {
            "facerec": (0, 0),
            "face++": (0, 0),
        }
        for i in tqdm.tqdm(range(total_batch)):
            iter_imgs1_path = imgs1_path[
                i * self.args.batch_size : (i + 1) * self.args.batch_size
            ]
            imgs1 = self._load_imgs(iter_imgs1_path)
            anchors = self.__find_best_anchor(imgs1, anchor_imgs)
            result = self.effectiveness.calculate_single_effectiveness(imgs1, anchors)
            self.__merge_dict(effectivenesses, result)
            tqdm.tqdm.write(
                f"Face Recognition/Face++: {result['facerec'][0]/result['facerec'][1]*100:.3f}/{result['facerec'][1]}, {result['face++'][0]/result['face++'][1]*100:.3f}/{result['face++'][1]}"
            )

        self.logger.info(
            f"Face Recognition/Face++: {effectivenesses['facerec'][0]/effectivenesses['facerec'][1]*100:.3f}/{effectivenesses['facerec'][1]}, {effectivenesses['face++'][0]/effectivenesses['face++'][1]*100:.3f}/{effectivenesses['face++'][1]}"
        )
