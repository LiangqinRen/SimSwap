from common_base import Base

import os
import io
import random
import torch

import torch.nn as nn
import torch.nn.functional as F

from os.path import join
from torchvision.utils import save_image
from tqdm import tqdm
from torch import tensor
from torchvision.transforms import ToPILImage, ToTensor
from PIL import Image


class Robustness(Base, nn.Module):
    def __init__(self, args, logger):
        super().__init__(args, logger)

        self.test_imgs_path = self.__get_all_test_imgs_path()

    def _load_logo(self) -> tensor:
        logo_path = join(self.args.data_dir, "samples", "usenix.png")
        logo = self._load_imgs([logo_path])
        return logo

    def _gauss_noise(self, pert: tensor, gauss_mean: float, gauss_std: float) -> tensor:
        gauss_noise = gauss_mean + gauss_std * torch.randn(pert.shape).cuda()
        noise_pert = pert + gauss_noise

        return noise_pert

    def _webp_compress(self, imgs, quality):
        compressed_imgs = []
        for i in range(imgs.size(0)):
            img = imgs[i]
            pil_img = ToPILImage()(img)

            buffer = io.BytesIO()
            pil_img.save(buffer, format="WEBP", quality=quality)
            buffer.seek(0)

            compressed_img = Image.open(buffer)
            compressed_img = ToTensor()(compressed_img)

            compressed_imgs.append(compressed_img)

        return torch.stack(compressed_imgs).cuda()

    def _crop(self, imgs: tensor, thickness: float) -> tensor:
        crop_imgs = imgs.clone()
        crop_imgs[:, :, :thickness, :] = 0
        crop_imgs[:, :, -thickness:, :] = 0
        crop_imgs[:, :, :, :thickness] = 0
        crop_imgs[:, :, :, -thickness:] = 0

        return crop_imgs

    def _logo(self, imgs: tensor, logo: tensor) -> tensor:
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

    def _brightness(self, imgs, brightness_factor: float):
        adjusted_tensor = imgs * brightness_factor
        adjusted_tensor = torch.clamp(adjusted_tensor, 0, 1)

        return adjusted_tensor

    def __get_all_test_imgs_path(self) -> list:
        imgs_path = []
        for people in os.listdir(
            join(os.path.dirname(os.path.abspath(__file__)), self.args.data_dir, "test")
        ):
            for img_name in os.listdir(
                join(
                    os.path.dirname(os.path.abspath(__file__)),
                    self.args.data_dir,
                    "test",
                    people,
                )
            ):
                imgs_path.append(
                    join(
                        os.path.dirname(os.path.abspath(__file__)),
                        self.args.data_dir,
                        "test",
                        people,
                        img_name,
                    )
                )

        return imgs_path

    def artificial_gan_fingerprints_swap(self):
        fingerprints_path = sorted(
            os.listdir(join(self.args.data_dir, "fingerprinted_images"))
        )
        fingerprints_path = [
            join(self.args.data_dir, "fingerprinted_images", i)
            for i in fingerprints_path
        ]
        logo = self._load_logo()

        total_batch = len(fingerprints_path) // self.args.batch_size
        os.makedirs(join(self.args.log_dir, "image", "noise"), exist_ok=True)
        os.makedirs(join(self.args.log_dir, "image", "compress"), exist_ok=True)
        os.makedirs(join(self.args.log_dir, "image", "crop"), exist_ok=True)
        os.makedirs(join(self.args.log_dir, "image", "logo"), exist_ok=True)
        os.makedirs(join(self.args.log_dir, "image", "inc_bright"), exist_ok=True)
        os.makedirs(join(self.args.log_dir, "image", "dec_bright"), exist_ok=True)
        for i in tqdm(range(total_batch)):
            iter_imgs1_path = fingerprints_path[
                i * self.args.batch_size : (i + 1) * self.args.batch_size
            ]

            imgs1 = self._load_imgs(iter_imgs1_path)

            noise_imgs1 = self._gauss_noise(imgs1, 0, 0.1)
            compress_imgs1 = self._webp_compress(imgs1, 80)
            crop_imgs1 = self._crop(imgs1, 20)
            logo_imgs1 = self._logo(imgs1, logo)
            inc_bright_imgs1 = self._brightness(imgs1, 1.25)
            dec_bright_imgs1 = self._brightness(imgs1, 0.75)

            for j in range(noise_imgs1.shape[0]):
                save_image(
                    noise_imgs1[j],
                    join(self.args.log_dir, "image", "noise", f"{i}_{j}.png"),
                )
            for j in range(compress_imgs1.shape[0]):
                save_image(
                    compress_imgs1[j],
                    join(self.args.log_dir, "image", "compress", f"{i}_{j}.png"),
                )
            for j in range(crop_imgs1.shape[0]):
                save_image(
                    crop_imgs1[j],
                    join(self.args.log_dir, "image", "crop", f"{i}_{j}.png"),
                )
            for j in range(logo_imgs1.shape[0]):
                save_image(
                    logo_imgs1[j],
                    join(self.args.log_dir, "image", "logo", f"{i}_{j}.png"),
                )
            for j in range(inc_bright_imgs1.shape[0]):
                save_image(
                    inc_bright_imgs1[j],
                    join(self.args.log_dir, "image", "inc_bright", f"{i}_{j}.png"),
                )
            for j in range(dec_bright_imgs1.shape[0]):
                save_image(
                    dec_bright_imgs1[j],
                    join(self.args.log_dir, "image", "dec_bright", f"{i}_{j}.png"),
                )

    def sepmark(self, img1: tensor) -> dict:
        logo = self._load_logo()

        noise_img1 = self._gauss_noise(img1, 0, 0.1)
        compress_img1 = self._webp_compress(img1, 80)
        crop_img1 = self._crop(img1, 20)
        logo_img1 = self._logo(img1, logo)
        inc_bright_img1 = self._brightness(img1, 1.25)
        dec_bright_img1 = self._brightness(img1, 0.75)

        img2_path = random.sample(self.test_imgs_path, 1)
        img2 = self._load_imgs(img2_path)

        img1_noise_identity = self._get_imgs_identity(noise_img1)
        img1_noise_src_swap = self.target(None, img2, img1_noise_identity, None, True)

        img1_compress_identity = self._get_imgs_identity(compress_img1)
        img1_compress_src_swap = self.target(
            None, img2, img1_compress_identity, None, True
        )

        img1_crop_identity = self._get_imgs_identity(crop_img1)
        img1_crop_src_swap = self.target(None, img2, img1_crop_identity, None, True)

        img1_logo_identity = self._get_imgs_identity(logo_img1)
        img1_logo_src_swap = self.target(None, img2, img1_logo_identity, None, True)

        img1_inc_bright_identity = self._get_imgs_identity(inc_bright_img1)
        img1_inc_bright_src_swap = self.target(
            None, img2, img1_inc_bright_identity, None, True
        )

        img1_dec_bright_identity = self._get_imgs_identity(dec_bright_img1)
        img1_dec_bright_src_swap = self.target(
            None, img2, img1_dec_bright_identity, None, True
        )

        img2_identity = self._get_imgs_identity(img2)
        img1_noise_tgt_swap = self.target(None, noise_img1, img2_identity, None, True)
        img1_compress_tgt_swap = self.target(
            None, compress_img1, img2_identity, None, True
        )
        img1_crop_tgt_swap = self.target(None, crop_img1, img2_identity, None, True)
        img1_logo_tgt_swap = self.target(None, logo_img1, img2_identity, None, True)
        img1_inc_bright_tgt_swap = self.target(
            None, inc_bright_img1, img2_identity, None, True
        )
        img1_dec_bright_tgt_swap = self.target(
            None, dec_bright_img1, img2_identity, None, True
        )

        img1_identity = self._get_imgs_identity(img1)
        img1_src_swap = self.target(None, img2, img1_identity, None, True)
        img1_tgt_swap = self.target(None, img1, img2_identity, None, True)

        results = {
            "img1_src_swap": img1_src_swap,
            "img1_tgt_swap": img1_tgt_swap,
            "img1_noise_src_swap": img1_noise_src_swap,
            "img1_compress_src_swap": img1_compress_src_swap,
            "img1_crop_src_swap": img1_crop_src_swap,
            "img1_logo_src_swap": img1_logo_src_swap,
            "img1_inc_bright_src_swap": img1_inc_bright_src_swap,
            "img1_dec_bright_src_swap": img1_dec_bright_src_swap,
            "img1_noise_tgt_swap": img1_noise_tgt_swap,
            "img1_compress_tgt_swap": img1_compress_tgt_swap,
            "img1_crop_tgt_swap": img1_crop_tgt_swap,
            "img1_logo_tgt_swap": img1_logo_tgt_swap,
            "img1_inc_bright_tgt_swap": img1_inc_bright_tgt_swap,
            "img1_dec_bright_tgt_swap": img1_dec_bright_tgt_swap,
        }

        return results


def main(args, logger):
    robustness = Robustness(args, logger)
    robustness.artificial_gan_fingerprints_swap()
