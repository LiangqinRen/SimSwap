from common_base import Base

import os
import random
import cv2
import torch

import numpy as np
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

        return blurred_pert

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

    def artificial_gan_fingerprints_swap(self):
        fingerprints_path = sorted(
            os.listdir(join(self.args.data_dir, "fingerprinted_test1"))
        )
        fingerprints_path = [
            join(self.args.data_dir, "fingerprinted_test1", i)
            for i in fingerprints_path
        ]
        imgs2_path = sorted(os.listdir(join(self.args.data_dir, "test2")))
        imgs2_path = [join(self.args.data_dir, "test2", i) for i in imgs2_path]

        total_batch = (
            min(len(fingerprints_path), len(imgs2_path)) // self.args.batch_size
        )
        os.makedirs(join(self.args.log_dir, "image", "noise"), exist_ok=True)
        os.makedirs(join(self.args.log_dir, "image", "compress"), exist_ok=True)
        os.makedirs(join(self.args.log_dir, "image", "rotate"), exist_ok=True)
        for i in tqdm(range(total_batch)):
            iter_imgs1_path = fingerprints_path[
                i * self.args.batch_size : (i + 1) * self.args.batch_size
            ]
            iter_imgs2_path = imgs2_path[
                i * self.args.batch_size : (i + 1) * self.args.batch_size
            ]

            imgs1 = self._load_imgs(iter_imgs1_path)
            imgs2 = self._load_imgs(iter_imgs1_path)
            imgs2_identity = self._get_imgs_identity(imgs2)

            noise_imgs1 = self.__gauss_noise(imgs1, 0, 0.1)
            noise_imgs1 = self.target(None, noise_imgs1, imgs2_identity, None, True)
            compress_imgs1 = self.__jpeg_compress(imgs1, 90)
            compress_imgs1 = self.target(
                None, compress_imgs1, imgs2_identity, None, True
            )
            rotate_imgs1 = self.__rotate(imgs1, 60)
            rotate_imgs1 = self.target(None, rotate_imgs1, imgs2_identity, None, True)

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

    def sepmark(self, img1: tensor) -> dict:
        gauss_mean, gauss_std = 0, 0.1
        gauss_size, gauss_sigma = 5, 3.0
        jpeg_ratio = 75
        rotate_angle = 60

        noise_img1 = self.__gauss_noise(img1, gauss_mean, gauss_std)
        blur_img1 = self.__gauss_blur(img1, gauss_size, gauss_sigma)
        compress_img1 = self.__jpeg_compress(img1, jpeg_ratio)
        rotate_img1 = self.__rotate(img1, rotate_angle)

        img2_path = random.sample(self.test_imgs_path, 1)
        img2 = self._load_imgs(img2_path)

        img1_noise_identity = self._get_imgs_identity(noise_img1)
        img1_noise_src_swap = self.target(None, img2, img1_noise_identity, None, True)

        img1_blur_identity = self._get_imgs_identity(blur_img1)
        img1_blur_src_swap = self.target(None, img2, img1_blur_identity, None, True)
        img1_compress_identity = self._get_imgs_identity(compress_img1)
        img1_compress_src_swap = self.target(
            None, img2, img1_compress_identity, None, True
        )
        img1_rotate_identity = self._get_imgs_identity(rotate_img1)
        img1_rotate_src_swap = self.target(None, img2, img1_rotate_identity, None, True)

        img2_identity = self._get_imgs_identity(img2)
        img1_blur_tgt_swap = self.target(None, blur_img1, img2_identity, None, True)
        img1_noise_tgt_swap = self.target(None, noise_img1, img2_identity, None, True)
        img1_compress_tgt_swap = self.target(
            None, compress_img1, img2_identity, None, True
        )
        img1_rotate_tgt_swap = self.target(None, rotate_img1, img2_identity, None, True)

        img1_identity = self._get_imgs_identity(img1)
        img1_src_swap = self.target(None, img2, img1_identity, None, True)
        img1_tgt_swap = self.target(None, img1, img2_identity, None, True)

        results = {
            "img1_noise_src_swap": img1_noise_src_swap,
            "img1_src_swap": img1_src_swap,
            "img1_tgt_swap": img1_tgt_swap,
            "img1_blur_src_swap": img1_blur_src_swap,
            "img1_compress_src_swap": img1_compress_src_swap,
            "img1_rotate_src_swap": img1_rotate_src_swap,
            "img1_blur_tgt_swap": img1_blur_tgt_swap,
            "img1_noise_tgt_swap": img1_noise_tgt_swap,
            "img1_compress_tgt_swap": img1_compress_tgt_swap,
            "img1_rotate_tgt_swap": img1_rotate_tgt_swap,
        }

        return results


def main(args, logger):
    robustness = Robustness(args, logger)
    robustness.sepmark()
