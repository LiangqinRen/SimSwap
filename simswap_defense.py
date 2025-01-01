from common_base import Base

import os
import io
import random
import torch
import math

import torch.nn as nn
import torch.optim as optim


from os.path import join
from torchvision.utils import save_image
from models.fs_networks import Generator
from torch import tensor
from torchvision.transforms import ToPILImage, ToTensor
from PIL import Image


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

        self.pgd_rgb_limits = {"R": 0.075, "G": 0.03, "B": 0.075}
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
            content += f"{tuple(f'{v[0]/v[1]*100:.3f}/{v[1]:.0f}' for _,v in effectiveness[effec].items())} "

        return content

    def __generate_summary_effectiveness_log(self, data: dict, item: str) -> str:
        content = ""
        for effec in data[item]:
            content += f"{tuple(f'{v[0]/v[1]*100:.3f}/{v[1]:.0f}' for _,v in data[item][effec].items())} "

        return content

    def __generate_iter_robustness_log(self, source: dict, target: dict) -> str:
        content = ""
        for effec in source:
            content += f"{tuple(f'{v[0]/v[1]*100:.3f}/{v[1]:.0f}' for _,v in source[effec].items())} "
        for effec in target:
            content += f"{tuple(f'{v[0]/v[1]*100:.3f}/{v[1]:.0f}' for _,v in target[effec].items())} "
        return content

    def __generate_summary_robustness_log(self, data: dict) -> str:
        content = ""
        source = data["pert_as_src_effectiveness"]
        target = data["pert_as_tgt_effectiveness"]
        for effec in source:
            content += f"{tuple(f'{v[0]/v[1]*100:.3f}/{v[1]:.0f}' for _,v in source[effec].items())} "
        for effec in target:
            content += f"{tuple(f'{v[0]/v[1]*100:.3f}/{v[1]:.0f}' for _,v in target[effec].items())} "
        return content

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

    def __load_logo(self) -> tensor:
        logo_path = join(self.samples_dir, "ccs.png")
        logo = self._load_imgs([logo_path])
        return logo

    def __perturb_pgd_imgs(
        self, imgs: tensor, best_anchor_imgs: tensor, silent: bool = False
    ) -> tuple[tensor, tensor]:
        l2_loss = nn.MSELoss().cuda()
        best_anchor_identity = self._get_imgs_identity(best_anchor_imgs)
        imgs_latent_code = self.target.netG.encoder(imgs)
        x_imgs = imgs.clone().detach()
        epsilon = self.args.pgd_epsilon * (torch.max(imgs) - torch.min(imgs)) / 2
        limits = (
            torch.tensor(
                [
                    self.pgd_rgb_limits["R"],
                    self.pgd_rgb_limits["G"],
                    self.pgd_rgb_limits["B"],
                ]
            )
            .view(1, 3, 1, 1)
            .cuda()
        )
        best_imgs, best_loss = None, float("inf")
        for epoch in range(self.args.epochs):
            x_imgs.requires_grad = True

            x_identity = self._get_imgs_identity(x_imgs)
            identity_diff_loss = l2_loss(x_identity, best_anchor_identity.detach())
            x_latent_code = self.target.netG.encoder(x_imgs)

            pert_diff_loss = l2_loss(x_imgs, imgs.detach())
            identity_diff_loss = l2_loss(x_identity, best_anchor_identity.detach())
            latent_code_diff_loss = -torch.clamp(
                l2_loss(x_latent_code, imgs_latent_code.detach()),
                0,
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
                min=imgs - limits,
                max=imgs + limits,
            )
            x_imgs = torch.clamp(x_imgs, 0, 1)

            if loss < best_loss:
                best_loss = loss
                best_imgs = x_imgs

            if not silent:
                self.logger.info(
                    f"[Epoch {epoch+1:4}/{self.args.epochs:4}]loss: {loss:.5f}({self.pgd_loss_weights['pert'] * pert_diff_loss.item():.5f}, {self.pgd_loss_weights['identity'] * identity_diff_loss.item():.5f}, {self.pgd_loss_weights['latent'] * latent_code_diff_loss.item():.5f})"
                )

        return best_imgs

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
            f"loss_weights: {self.pgd_loss_weights}, loss_limits: {self.pgd_loss_limits}, RGB_limits: {self.pgd_rgb_limits}"
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

        random.shuffle(source_imgs_path)
        random.shuffle(target_imgs_path)

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

        accumulate_anchor_distance = []
        accumulate_pert_swap_distance = []
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

            anchor_distance = self.effectiveness.get_images_distance(
                imgs1, best_anchor_imgs
            )
            pert_swap_distance = self.effectiveness.get_images_distance(
                imgs1, pert_imgs1_src_swap
            )
            with open(join(self.args.log_dir, "anchor_distances.txt"), "a") as f:
                for dist in anchor_distance:
                    f.write(f"{dist}\n")
            with open(join(self.args.log_dir, "pert_swap_distances.txt"), "a") as f:
                for dist in pert_swap_distance:
                    f.write(f"{dist}\n")
            accumulate_anchor_distance.extend(anchor_distance)
            accumulate_pert_swap_distance.extend(pert_swap_distance)

            del imgs1, imgs2, x_imgs, best_anchor_imgs
            del imgs1_src_swap, pert_imgs1_src_swap, imgs1_tgt_swap, pert_imgs1_tgt_swap
            torch.cuda.empty_cache()

            self.logger.info(
                f"""
            utility(mse, psnr, ssim, lpips), effectiveness{self.effectiveness.candi_funcs.keys()} source(pert, swap, pert_swap, anchor) target(swap, pert_swap)
            anchor distances: {sum(anchor_distance)/len(anchor_distance):.3f}
            pert swap distances: {sum(pert_swap_distance)/len(pert_swap_distance):.3f}
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
            {sum(accumulate_anchor_distance)/len(accumulate_anchor_distance):.3f}
            {sum(accumulate_pert_swap_distance)/len(accumulate_pert_swap_distance):.3f}
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

    def __webp_compress(self, imgs, quality):
        compressed_tensors = []

        for i in range(imgs.size(0)):
            single_image_tensor = imgs[i]

            pil_image = ToPILImage()(single_image_tensor)

            buffer = io.BytesIO()
            pil_image.save(buffer, format="WEBP", quality=quality)
            buffer.seek(0)

            compressed_image = Image.open(buffer)
            compressed_tensor = ToTensor()(compressed_image)

            compressed_tensors.append(compressed_tensor)

        return torch.stack(compressed_tensors).cuda()

    def __crop(self, imgs: tensor, thickness: float) -> tensor:
        crop_imgs = imgs.clone()
        crop_imgs[:, :, :thickness, :] = 0
        crop_imgs[:, :, -thickness:, :] = 0
        crop_imgs[:, :, :, :thickness] = 0
        crop_imgs[:, :, :, -thickness:] = 0

        return crop_imgs

    def __brightness(self, imgs, brightness_factor: float):
        adjusted_tensor = imgs * brightness_factor
        adjusted_tensor = torch.clamp(adjusted_tensor, 0, 1)

        return adjusted_tensor

    def __logo(self, imgs: tensor, logo: tensor) -> tensor:
        alpha = 0.75
        _, _, img_height, img_width = imgs.shape
        _, _, logo_height, logo_width = logo.shape

        x_offset = img_width - logo_width
        y_offset = img_height - logo_height

        logo_imgs = imgs.clone()

        logo_imgs[
            :, :, y_offset : y_offset + logo_height, x_offset : x_offset + logo_width
        ] = (
            imgs[
                :,
                :,
                y_offset : y_offset + logo_height,
                x_offset : x_offset + logo_width,
            ]
            * (1 - alpha)
            + logo * alpha
        )

        return logo_imgs

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
            None,
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
        compress_rate = 80

        compress_imgs = self.__webp_compress(imgs1, compress_rate)
        compress_identity = self._get_imgs_identity(compress_imgs)
        compress_pert_imgs = self.__webp_compress(x_imgs, compress_rate)
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
            None,
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

    def __get_brightness_metrics(
        self,
        imgs1: tensor,
        imgs2: tensor,
        x_imgs: tensor,
        anchors_imgs: tensor,
        factor: float,
    ) -> tuple[dict, dict, dict, dict]:
        brightness_imgs = self.__brightness(imgs1, factor)
        brightness_identity = self._get_imgs_identity(brightness_imgs)
        brightness_pert_imgs = self.__brightness(x_imgs, factor)
        brightness_pert_identity = self._get_imgs_identity(brightness_pert_imgs)

        imgs2_identity = self._get_imgs_identity(imgs2)
        brightness_swap_imgs = self.target(None, imgs2, brightness_identity, None, True)
        reverse_brightness_swap_imgs = self.target(
            None, brightness_imgs, imgs2_identity, None, True
        )
        brightness_pert_swap_imgs = self.target(
            None, imgs2, brightness_pert_identity, None, True
        )
        reverse_brightness_pert_swap_imgs = self.target(
            None, brightness_pert_imgs, imgs2_identity, None, True
        )

        source_effectivenesses = self.effectiveness.calculate_effectiveness(
            imgs1,
            None,
            brightness_swap_imgs,
            brightness_pert_swap_imgs,
            anchors_imgs,
        )
        target_effectivenesses = self.effectiveness.calculate_effectiveness(
            imgs2,
            None,
            reverse_brightness_swap_imgs,
            reverse_brightness_pert_swap_imgs,
            None,
        )

        self.__save_robustness_samples(
            f"brightness_{factor}",
            [
                imgs1,
                imgs2,
                brightness_imgs,
                brightness_swap_imgs,
                reverse_brightness_swap_imgs,
                x_imgs,
                brightness_pert_imgs,
                brightness_pert_swap_imgs,
                reverse_brightness_pert_swap_imgs,
            ],
        )

        return (
            source_effectivenesses,
            target_effectivenesses,
        )

    def __get_crop_metrics(
        self,
        imgs1: tensor,
        imgs2: tensor,
        x_imgs: tensor,
        anchors_imgs: tensor,
    ) -> tuple[dict, dict]:
        border_thickness = 20

        crop_imgs = self.__crop(imgs1, border_thickness)
        crop_identity = self._get_imgs_identity(crop_imgs)
        crop_pert_imgs = self.__crop(x_imgs, border_thickness)
        crop_pert_identity = self._get_imgs_identity(crop_pert_imgs)

        imgs2_identity = self._get_imgs_identity(imgs2)
        crop_swap_imgs = self.target(None, imgs2, crop_identity, None, True)
        reverse_crop_swap_imgs = self.target(
            None, crop_imgs, imgs2_identity, None, True
        )
        crop_pert_swap_imgs = self.target(None, imgs2, crop_pert_identity, None, True)
        reverse_crop_pert_swap_imgs = self.target(
            None, crop_pert_imgs, imgs2_identity, None, True
        )

        source_effectivenesses = self.effectiveness.calculate_effectiveness(
            imgs1,
            None,
            crop_swap_imgs,
            crop_pert_swap_imgs,
            anchors_imgs,
        )
        target_effectivenesses = self.effectiveness.calculate_effectiveness(
            imgs2, None, reverse_crop_swap_imgs, reverse_crop_pert_swap_imgs, None
        )

        self.__save_robustness_samples(
            "crop",
            [
                imgs1,
                imgs2,
                crop_imgs,
                crop_swap_imgs,
                reverse_crop_swap_imgs,
                x_imgs,
                crop_pert_imgs,
                crop_pert_swap_imgs,
                reverse_crop_pert_swap_imgs,
            ],
        )

        return (
            source_effectivenesses,
            target_effectivenesses,
        )

    def __get_blocking_metrics(
        self,
        imgs1: tensor,
        imgs2: tensor,
        x_imgs: tensor,
        anchors_imgs: tensor,
    ) -> tuple[dict, dict]:
        cover_imgs = self.__crop(imgs1)
        cover_identity = self._get_imgs_identity(cover_imgs)
        cover_pert_imgs = self.__crop(x_imgs)
        cover_pert_identity = self._get_imgs_identity(cover_pert_imgs)

        imgs2_identity = self._get_imgs_identity(imgs2)
        cover_swap_imgs = self.target(None, imgs2, cover_identity, None, True)
        reverse_cover_swap_imgs = self.target(
            None, cover_imgs, imgs2_identity, None, True
        )
        cover_pert_swap_imgs = self.target(None, imgs2, cover_pert_identity, None, True)
        reverse_cover_pert_swap_imgs = self.target(
            None, cover_pert_imgs, imgs2_identity, None, True
        )

        source_effectivenesses = self.effectiveness.calculate_effectiveness(
            imgs1,
            None,
            cover_swap_imgs,
            cover_pert_swap_imgs,
            anchors_imgs,
        )
        target_effectivenesses = self.effectiveness.calculate_effectiveness(
            imgs2, None, reverse_cover_swap_imgs, reverse_cover_pert_swap_imgs, None
        )

        self.__save_robustness_samples(
            "cover",
            [
                imgs1,
                imgs2,
                cover_imgs,
                cover_swap_imgs,
                reverse_cover_swap_imgs,
                x_imgs,
                cover_pert_imgs,
                cover_pert_swap_imgs,
                reverse_cover_pert_swap_imgs,
            ],
        )

        return (
            source_effectivenesses,
            target_effectivenesses,
        )

    def __get_logo_metrics(
        self,
        imgs1: tensor,
        imgs2: tensor,
        x_imgs: tensor,
        logo: tensor,
        anchors_imgs: tensor,
    ) -> tuple[dict, dict]:
        logo_imgs = self.__logo(imgs1, logo)
        logo_identity = self._get_imgs_identity(logo_imgs)
        logo_pert_imgs = self.__logo(x_imgs, logo)
        logo_pert_identity = self._get_imgs_identity(logo_pert_imgs)

        imgs2_identity = self._get_imgs_identity(imgs2)
        logo_swap_imgs = self.target(None, imgs2, logo_identity, None, True)
        reverse_logo_swap_imgs = self.target(
            None, logo_imgs, imgs2_identity, None, True
        )
        logo_pert_swap_imgs = self.target(None, imgs2, logo_pert_identity, None, True)
        reverse_logo_pert_swap_imgs = self.target(
            None, logo_pert_imgs, imgs2_identity, None, True
        )

        source_effectivenesses = self.effectiveness.calculate_effectiveness(
            imgs1,
            None,
            logo_swap_imgs,
            logo_pert_swap_imgs,
            anchors_imgs,
        )
        target_effectivenesses = self.effectiveness.calculate_effectiveness(
            imgs2, None, reverse_logo_swap_imgs, reverse_logo_pert_swap_imgs, None
        )

        self.__save_robustness_samples(
            "logo",
            [
                imgs1,
                imgs2,
                logo_imgs,
                logo_swap_imgs,
                reverse_logo_swap_imgs,
                x_imgs,
                logo_pert_imgs,
                logo_pert_swap_imgs,
                reverse_logo_pert_swap_imgs,
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

        logo = self.__load_logo()

        (
            noise_source_effectivenesses,
            noise_target_effectivenesses,
        ) = self.__get_gauss_noise_metrics(imgs1, imgs2, x_imgs, best_anchor_imgs)

        (
            compress_source_effectivenesses,
            compress_target_effectivenesses,
        ) = self.__get_compress_metrics(imgs1, imgs2, x_imgs, best_anchor_imgs)

        (
            crop_source_effectivenesses,
            crop_target_effectivenesses,
        ) = self.__get_crop_metrics(imgs1, imgs2, x_imgs, best_anchor_imgs)

        (
            logo_source_effectivenesses,
            logo_target_effectivenesses,
        ) = self.__get_logo_metrics(imgs1, imgs2, x_imgs, logo, best_anchor_imgs)

        (
            inc_bright_source_effectivenesses,
            inc_bright_target_effectivenesses,
        ) = self.__get_brightness_metrics(imgs1, imgs2, x_imgs, best_anchor_imgs, 1.25)

        (
            dec_bright_source_effectivenesses,
            dec_bright_target_effectivenesses,
        ) = self.__get_brightness_metrics(imgs1, imgs2, x_imgs, best_anchor_imgs, 0.75)

        torch.cuda.empty_cache()
        self.logger.info(
            f"""
            noise, compress, crop, overlay, increase and decrease the brightness {self.effectiveness.candi_funcs.keys()}
            source(robust swap, robust pert swap), target(robust swap, robust pert swap)
            {self.__generate_iter_robustness_log(noise_source_effectivenesses,noise_target_effectivenesses)}
            {self.__generate_iter_robustness_log(compress_source_effectivenesses,compress_target_effectivenesses)}
            {self.__generate_iter_robustness_log(crop_source_effectivenesses,crop_target_effectivenesses)}
            {self.__generate_iter_robustness_log(logo_source_effectivenesses,logo_target_effectivenesses)}
            {self.__generate_iter_robustness_log(inc_bright_source_effectivenesses,inc_bright_target_effectivenesses)}
            {self.__generate_iter_robustness_log(dec_bright_source_effectivenesses,dec_bright_target_effectivenesses)}
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
        data = {
            "pert_as_src_effectiveness": {},
            "pert_as_tgt_effectiveness": {},
        }

        for effec in self.effectiveness.candi_funcs.keys():
            data["pert_as_src_effectiveness"][effec] = {
                "swap": (0, 0),
                "pert_swap": (0, 0),
                "anchor": (0, 0),
            }
            data["pert_as_tgt_effectiveness"][effec] = {
                "swap": (0, 0),
                "pert_swap": (0, 0),
            }
        from copy import deepcopy

        data = {
            "noise": deepcopy(data),
            "compress": deepcopy(data),
            "crop": deepcopy(data),
            "logo": deepcopy(data),
            "inc_bright": deepcopy(data),
            "dec_bright": deepcopy(data),
        }

        logo = self.__load_logo()
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
                crop_source_effectivenesses,
                crop_target_effectivenesses,
            ) = self.__get_crop_metrics(imgs1, imgs2, x_imgs, best_anchor_imgs)
            self.__merge_robustness_metric(
                data,
                crop_source_effectivenesses,
                crop_target_effectivenesses,
                "crop",
            )

            (
                logo_source_effectivenesses,
                logo_target_effectivenesses,
            ) = self.__get_logo_metrics(imgs1, imgs2, x_imgs, logo, best_anchor_imgs)
            self.__merge_robustness_metric(
                data,
                logo_source_effectivenesses,
                logo_target_effectivenesses,
                "logo",
            )

            (
                inc_bright_source_effectivenesses,
                inc_bright_target_effectivenesses,
            ) = self.__get_brightness_metrics(
                imgs1, imgs2, x_imgs, best_anchor_imgs, 1.25
            )
            self.__merge_robustness_metric(
                data,
                inc_bright_source_effectivenesses,
                inc_bright_target_effectivenesses,
                "inc_bright",
            )

            (
                dec_bright_source_effectivenesses,
                dec_bright_target_effectivenesses,
            ) = self.__get_brightness_metrics(
                imgs1, imgs2, x_imgs, best_anchor_imgs, 0.75
            )
            self.__merge_robustness_metric(
                data,
                dec_bright_source_effectivenesses,
                dec_bright_target_effectivenesses,
                "dec_bright",
            )

            torch.cuda.empty_cache()
            self.logger.info(
                f"""
            noise, compress, crop, overlay, increase and decrease the brightness {self.effectiveness.candi_funcs.keys()}
            source(robust swap, robust pert swap, anchor), target(robust swap, robust pert swap)
            {self.__generate_iter_robustness_log(noise_source_effectivenesses,noise_target_effectivenesses)}
            {self.__generate_iter_robustness_log(compress_source_effectivenesses,compress_target_effectivenesses)}
            {self.__generate_iter_robustness_log(crop_source_effectivenesses,crop_target_effectivenesses)}
            {self.__generate_iter_robustness_log(logo_source_effectivenesses,logo_target_effectivenesses)}
            {self.__generate_iter_robustness_log(inc_bright_source_effectivenesses,inc_bright_target_effectivenesses)}
            {self.__generate_iter_robustness_log(dec_bright_source_effectivenesses,dec_bright_target_effectivenesses)}
            """
            )

            self.logger.info(
                f"""[{i + 1}/{total_batch}]Average of {self.args.batch_size * (i + 1)} pictures
            {self.__generate_summary_robustness_log(data['noise'])}
            {self.__generate_summary_robustness_log(data['compress'])}
            {self.__generate_summary_robustness_log(data['crop'])}
            {self.__generate_summary_robustness_log(data['logo'])}
            {self.__generate_summary_robustness_log(data['inc_bright'])}
            {self.__generate_summary_robustness_log(data['dec_bright'])}
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

        self.logger.info(
            f"""
        utility(mse, psnr, ssim, lpips), effectiveness{self.effectiveness.candi_funcs.keys()} source(pert, swap, pert_swap) target(swap, pert_swap)
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

            self.logger.info(
                f"""
            utility(mse, psnr, ssim, lpips), effectiveness{self.effectiveness.candi_funcs.keys()} source(pert, swap, pert_swap) target(swap, pert_swap)
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

        logo = self.__load_logo()

        (
            noise_source_effectivenesses,
            noise_target_effectivenesses,
        ) = self.__get_gauss_noise_metrics(imgs1, imgs2, pert_imgs1, None)

        (
            compress_source_effectivenesses,
            compress_target_effectivenesses,
        ) = self.__get_compress_metrics(imgs1, imgs2, pert_imgs1, None)

        (
            crop_source_effectivenesses,
            crop_target_effectivenesses,
        ) = self.__get_crop_metrics(imgs1, imgs2, pert_imgs1, None)

        (
            logo_source_effectivenesses,
            logo_target_effectivenesses,
        ) = self.__get_logo_metrics(imgs1, imgs2, pert_imgs1, logo, None)

        (
            inc_bright_source_effectivenesses,
            inc_bright_target_effectivenesses,
        ) = self.__get_brightness_metrics(imgs1, imgs2, pert_imgs1, None, 1.25)

        (
            dec_bright_source_effectivenesses,
            dec_bright_target_effectivenesses,
        ) = self.__get_brightness_metrics(imgs1, imgs2, pert_imgs1, None, 0.75)

        torch.cuda.empty_cache()
        self.logger.info(
            f"""
            noise, compress, crop, overlay, increase and decrease the brightness {self.effectiveness.candi_funcs.keys()}
            source(robust swap, robust pert swap), target(robust swap, robust pert swap)
            {self.__generate_iter_robustness_log(noise_source_effectivenesses,noise_target_effectivenesses)}
            {self.__generate_iter_robustness_log(compress_source_effectivenesses,compress_target_effectivenesses)}
            {self.__generate_iter_robustness_log(crop_source_effectivenesses,crop_target_effectivenesses)}
            {self.__generate_iter_robustness_log(logo_source_effectivenesses,logo_target_effectivenesses)}
            {self.__generate_iter_robustness_log(inc_bright_source_effectivenesses,inc_bright_target_effectivenesses)}
            {self.__generate_iter_robustness_log(dec_bright_source_effectivenesses,dec_bright_target_effectivenesses)}
            """
        )

    def gan_both_robustness_metric(self):
        model_path = join("checkpoints", self.args.gan_test_models)
        self.GAN_G.load_state_dict(torch.load(model_path)["GAN_G_state_dict"])

        self.target.cuda().eval()
        self.GAN_G.cuda().eval()

        imgs1_path, imgs2_imgs_path = self._get_split_test_imgs_path()
        data = {
            "pert_as_src_effectiveness": {},
            "pert_as_tgt_effectiveness": {},
        }

        for effec in self.effectiveness.candi_funcs.keys():
            data["pert_as_src_effectiveness"][effec] = {
                "swap": (0, 0),
                "pert_swap": (0, 0),
            }
            data["pert_as_tgt_effectiveness"][effec] = {
                "swap": (0, 0),
                "pert_swap": (0, 0),
            }
        from copy import deepcopy

        data = {
            "noise": deepcopy(data),
            "compress": deepcopy(data),
            "crop": deepcopy(data),
            "logo": deepcopy(data),
            "inc_bright": deepcopy(data),
            "dec_bright": deepcopy(data),
        }

        logo = self.__load_logo()
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
                crop_source_effectivenesses,
                crop_target_effectivenesses,
            ) = self.__get_crop_metrics(imgs1, imgs2, pert_imgs1, None)
            self.__merge_robustness_metric(
                data,
                crop_source_effectivenesses,
                crop_target_effectivenesses,
                "crop",
            )

            (
                logo_source_effectivenesses,
                logo_target_effectivenesses,
            ) = self.__get_logo_metrics(imgs1, imgs2, pert_imgs1, logo, None)
            self.__merge_robustness_metric(
                data,
                logo_source_effectivenesses,
                logo_target_effectivenesses,
                "logo",
            )

            (
                inc_bright_source_effectivenesses,
                inc_bright_target_effectivenesses,
            ) = self.__get_brightness_metrics(imgs1, imgs2, pert_imgs1, None, 1.25)
            self.__merge_robustness_metric(
                data,
                inc_bright_source_effectivenesses,
                inc_bright_target_effectivenesses,
                "inc_bright",
            )

            (
                dec_bright_source_effectivenesses,
                dec_bright_target_effectivenesses,
            ) = self.__get_brightness_metrics(imgs1, imgs2, pert_imgs1, None, 0.75)
            self.__merge_robustness_metric(
                data,
                dec_bright_source_effectivenesses,
                dec_bright_target_effectivenesses,
                "dec_bright",
            )

            torch.cuda.empty_cache()
            self.logger.info(
                f"""
            noise, compress, crop, overlay, increase and decrease the brightness {self.effectiveness.candi_funcs.keys()}
            source(robust swap, robust pert swap, anchor), target(robust swap, robust pert swap)
            {self.__generate_iter_robustness_log(noise_source_effectivenesses,noise_target_effectivenesses)}
            {self.__generate_iter_robustness_log(compress_source_effectivenesses,compress_target_effectivenesses)}
            {self.__generate_iter_robustness_log(crop_source_effectivenesses,crop_target_effectivenesses)}
            {self.__generate_iter_robustness_log(logo_source_effectivenesses,logo_target_effectivenesses)}
            {self.__generate_iter_robustness_log(inc_bright_source_effectivenesses,inc_bright_target_effectivenesses)}
            {self.__generate_iter_robustness_log(dec_bright_source_effectivenesses,dec_bright_target_effectivenesses)}
            """
            )

            self.logger.info(
                f"""[{i + 1}/{total_batch}]Average of {self.args.batch_size * (i + 1)} pictures
            {self.__generate_summary_robustness_log(data['noise'])}
            {self.__generate_summary_robustness_log(data['compress'])}
            {self.__generate_summary_robustness_log(data['crop'])}
            {self.__generate_summary_robustness_log(data['logo'])}
            {self.__generate_summary_robustness_log(data['inc_bright'])}
            {self.__generate_summary_robustness_log(data['dec_bright'])}
            """
            )

    def __calculate_anchor_distance(self, imgs: tensor, anchors: tensor) -> list[float]:
        imgs_ndarray = imgs.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255.0
        anchors_ndarray = anchors.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255.0

        distances = []
        for i in range(imgs.shape[0]):
            distance = self.effectiveness.get_image_distance(
                imgs_ndarray[i], anchors_ndarray[i]
            )
            if distance is math.nan:
                continue
            distances.append(distance)

        return distances

    def check_img_anchor_difference(self) -> None:
        imgs1_path, _ = self._get_split_test_imgs_path()
        total_batch = len(imgs1_path) // self.args.batch_size

        accumulate_distances = []
        accumulate_matching_count, accumulate_valid_count = 0, 0

        for batch in range(1, total_batch + 1):
            iter_imgs1_path = imgs1_path[
                (batch - 1) * self.args.batch_size : batch * self.args.batch_size
            ]
            imgs1 = self._load_imgs(iter_imgs1_path)
            anchors = self.anchor.find_best_anchors(imgs1)

            distances = self.__calculate_anchor_distance(imgs1, anchors)
            matching_count, valid_count = (
                self.effectiveness.calculate_single_effectiveness(imgs1, anchors)
            )

            with open(join(self.args.log_dir, "anchor_distances.txt"), "a") as f:
                for dist in distances:
                    f.write(f"{dist}\n")

            accumulate_distances.extend(distances)
            accumulate_matching_count += matching_count
            accumulate_valid_count += valid_count

            self.logger.info(
                f"min, max, avg distances are {min(distances):.3f}, {max(distances):.3f}, {sum(distances)/len(distances):.3f}, the facenet, face++ effectiveness are {sum([1 if dist <= self.args.effectiveness['facenet']['threshold'] else 0 for dist in distances])/len(distances)*100:.3f}, {matching_count/valid_count*100:.3f}"
            )
            self.logger.info(
                f"[{batch}/{total_batch}]Average of {self.args.batch_size * (batch)} pictures, min, max, avg distances are {min(accumulate_distances):.3f}, {max(accumulate_distances):.3f}, {sum(accumulate_distances)/len(accumulate_distances):.3f}, the facenet, face++ effectiveness are {sum([1 if dist <= self.args.effectiveness['facenet']['threshold'] else 0 for dist in accumulate_distances])/len(accumulate_distances)*100:.3f}, {accumulate_matching_count/accumulate_valid_count*100:.3f}"
            )
