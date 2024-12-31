from common_base import Base

import os
import random
import torch

import torch.nn as nn
import torch.nn.functional as F

from os.path import join
from torchvision.utils import save_image
from tqdm import tqdm
from torch import tensor


class Robustness(Base, nn.Module):
    def __init__(self, args, logger):
        super().__init__(args, logger)

        self.test_imgs_path = self.__get_all_test_imgs_path()

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

    def __gauss_noise(
        self, pert: tensor, gauss_mean: float, gauss_std: float
    ) -> tensor:
        gauss_noise = gauss_mean + gauss_std * torch.randn(pert.shape).cuda()
        noise_pert = pert + gauss_noise

        return noise_pert

    def __jpeg_compress(self, imgs: tensor, ratio: int) -> tensor:
        import torchvision.io as io

        compressed_imgs = []
        for img in imgs:
            compressed_bytes = io.encode_jpeg(
                torch.clip(img * 255, 0, 255).to(torch.uint8).cpu(), quality=ratio
            )
            decode_img = io.decode_jpeg(compressed_bytes)
            compressed_imgs.append(decode_img.float().cuda() / 255.0)

        return torch.stack(compressed_imgs)

    def __rotate(self, imgs: tensor, angle: float) -> None:
        import torchvision.transforms.functional as F
        from torchvision.transforms import InterpolationMode

        rotated_imgs = torch.stack(
            [
                F.rotate(tensor, angle, interpolation=InterpolationMode.BILINEAR)
                for tensor in imgs
            ]
        )
        restore_imgs = torch.stack(
            [
                F.rotate(tensor, -angle, interpolation=InterpolationMode.BILINEAR)
                for tensor in rotated_imgs
            ]
        )
        return restore_imgs

    def __crop(self, pert: tensor, ratio: float) -> tensor:
        from torchvision.transforms import CenterCrop

        crop_size = (int(224 * ratio / 100), int(224 * ratio / 100))
        center_crop = CenterCrop(size=crop_size)

        cropped_img = center_crop(pert)
        resized_image = F.interpolate(cropped_img, size=(224, 224), mode="bilinear")

        return resized_image

    def __cover(self, imgs: tensor) -> tensor:
        x, y, w, h = 174, 174, 50, 50
        cover_imgs = imgs.clone().detach()
        cover_imgs[:, :, y : y + h, x : x + w] = 0

        return cover_imgs

    def artificial_gan_fingerprints_swap(self):
        fingerprints_path = sorted(
            os.listdir(join(self.args.data_dir, "fingerprinted_images"))
        )
        fingerprints_path = [
            join(self.args.data_dir, "fingerprinted_images", i)
            for i in fingerprints_path
        ]

        total_batch = len(fingerprints_path) // self.args.batch_size
        os.makedirs(join(self.args.log_dir, "image", "noise"), exist_ok=True)
        os.makedirs(join(self.args.log_dir, "image", "compress"), exist_ok=True)
        os.makedirs(join(self.args.log_dir, "image", "rotate"), exist_ok=True)
        os.makedirs(join(self.args.log_dir, "image", "crop"), exist_ok=True)
        os.makedirs(join(self.args.log_dir, "image", "cover"), exist_ok=True)
        for i in tqdm(range(total_batch)):
            iter_imgs1_path = fingerprints_path[
                i * self.args.batch_size : (i + 1) * self.args.batch_size
            ]

            imgs1 = self._load_imgs(iter_imgs1_path)

            noise_imgs1 = self.__gauss_noise(imgs1, 0, 0.1)
            compress_imgs1 = self.__jpeg_compress(imgs1, 85)
            rotate_imgs1 = self.__rotate(imgs1, 60)
            crop_imgs1 = self.__crop(imgs1, 90)
            cover_imgs1 = self.__cover(imgs1)

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
            for j in range(rotate_imgs1.shape[0]):
                save_image(
                    rotate_imgs1[j],
                    join(self.args.log_dir, "image", "rotate", f"{i}_{j}.png"),
                )
            for j in range(crop_imgs1.shape[0]):
                save_image(
                    crop_imgs1[j],
                    join(self.args.log_dir, "image", "crop", f"{i}_{j}.png"),
                )
            for j in range(cover_imgs1.shape[0]):
                save_image(
                    cover_imgs1[j],
                    join(self.args.log_dir, "image", "cover", f"{i}_{j}.png"),
                )

    def sepmark(self, img1: tensor) -> dict:
        gauss_mean, gauss_std = 0, 0.1
        jpeg_ratio = 85
        rotate_angle = 60
        crop_ratio = 90

        noise_img1 = self.__gauss_noise(img1, gauss_mean, gauss_std)
        compress_img1 = self.__jpeg_compress(img1, jpeg_ratio)
        rotate_img1 = self.__rotate(img1, rotate_angle)
        crop_img1 = self.__crop(img1, crop_ratio)
        cover_img1 = self.__cover(img1)

        img2_path = random.sample(self.test_imgs_path, 1)
        img2 = self._load_imgs(img2_path)

        img1_noise_identity = self._get_imgs_identity(noise_img1)
        img1_noise_src_swap = self.target(None, img2, img1_noise_identity, None, True)

        img1_compress_identity = self._get_imgs_identity(compress_img1)
        img1_compress_src_swap = self.target(
            None, img2, img1_compress_identity, None, True
        )

        img1_rotate_identity = self._get_imgs_identity(rotate_img1)
        img1_rotate_src_swap = self.target(None, img2, img1_rotate_identity, None, True)

        img1_crop_identity = self._get_imgs_identity(crop_img1)
        img1_crop_src_swap = self.target(None, img2, img1_crop_identity, None, True)

        img1_cover_identity = self._get_imgs_identity(cover_img1)
        img1_cover_src_swap = self.target(None, img2, img1_cover_identity, None, True)

        img2_identity = self._get_imgs_identity(img2)
        img1_noise_tgt_swap = self.target(None, noise_img1, img2_identity, None, True)
        img1_compress_tgt_swap = self.target(
            None, compress_img1, img2_identity, None, True
        )
        img1_rotate_tgt_swap = self.target(None, rotate_img1, img2_identity, None, True)
        img1_crop_tgt_swap = self.target(None, crop_img1, img2_identity, None, True)
        img1_cover_tgt_swap = self.target(None, cover_img1, img2_identity, None, True)

        img1_identity = self._get_imgs_identity(img1)
        img1_src_swap = self.target(None, img2, img1_identity, None, True)
        img1_tgt_swap = self.target(None, img1, img2_identity, None, True)

        results = {
            "img1_src_swap": img1_src_swap,
            "img1_tgt_swap": img1_tgt_swap,
            "img1_noise_src_swap": img1_noise_src_swap,
            "img1_compress_src_swap": img1_compress_src_swap,
            "img1_rotate_src_swap": img1_rotate_src_swap,
            "img1_crop_src_swap": img1_crop_src_swap,
            "img1_cover_src_swap": img1_cover_src_swap,
            "img1_noise_tgt_swap": img1_noise_tgt_swap,
            "img1_compress_tgt_swap": img1_compress_tgt_swap,
            "img1_rotate_tgt_swap": img1_rotate_tgt_swap,
            "img1_crop_tgt_swap": img1_crop_tgt_swap,
            "img1_cover_tgt_swap": img1_cover_tgt_swap,
        }

        return results


def main(args, logger):
    robustness = Robustness(args, logger)
    robustness.artificial_gan_fingerprints_swap()
