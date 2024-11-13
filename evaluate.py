import cv2
import os
import torch
import lpips
import numpy as np
import math

from facenet_pytorch import MTCNN, InceptionResnetV1
from skimage import metrics
from torchvision.models import vgg16, VGG16_Weights
from torch import tensor


class Utility:
    import warnings

    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message="The parameter 'pretrained' is deprecated",
    )
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message="Arguments other than a weight enum or `None` for 'weights' are deprecated",
    )

    lpips_distance = lpips.LPIPS(net="vgg").cuda()

    def __init__(self):
        pass

    def compare(self, imgs1_list, imgs2_list):
        utilities = {"mse": [], "ssim": [], "psnr": []}
        for idx in range(min(imgs1_list.shape[0], imgs2_list.shape[0])):
            img1 = imgs1_list[idx]
            img2 = imgs2_list[idx]
            mse = metrics.mean_squared_error(img1, img2)
            utilities["mse"].append(mse)

            psnr = metrics.peak_signal_noise_ratio(img1, img2, data_range=255)
            utilities["psnr"].append(psnr)

            ssim = metrics.structural_similarity(
                img1, img2, channel_axis=2, multichannel=True, data_range=1
            )
            utilities["ssim"].append(ssim)

        for i in utilities:
            utilities[i] = np.mean(utilities[i])

        return (
            utilities["mse"],
            utilities["ssim"],
            utilities["psnr"],
        )

    def calculate_utility(self, imgs1: torch.tensor, imgs2: torch.tensor):
        utilities = {"mse": [], "psnr": [], "ssim": [], "lpips": []}

        imgs1_ndarray = imgs1.detach().cpu().numpy().transpose(0, 2, 3, 1)
        imgs2_ndarray = imgs2.detach().cpu().numpy().transpose(0, 2, 3, 1)
        for i in range(min(imgs1.shape[0], imgs2.shape[0])):
            mse = metrics.mean_squared_error(imgs1_ndarray[i], imgs2_ndarray[i])
            utilities["mse"].append(mse)

            psnr = metrics.peak_signal_noise_ratio(
                imgs1_ndarray[i], imgs2_ndarray[i], data_range=255
            )
            utilities["psnr"].append(psnr)

            ssim = metrics.structural_similarity(
                imgs1_ndarray[i],
                imgs2_ndarray[i],
                channel_axis=2,
                multichannel=True,
                data_range=1,
            )
            utilities["ssim"].append(ssim)

            lpips_score = self.lpips_distance(imgs1[i], imgs2[i])
            utilities["lpips"].append(lpips_score.detach().cpu().numpy())

        for i in utilities:
            utilities[i] = np.mean(utilities[i])

        return utilities


class Effectiveness:
    def __init__(self, threshold):
        self.threshold = threshold

        self.mtcnn = MTCNN(
            image_size=160,
            device="cuda",
            selection_method="largest",  # largest probability center_weighted_size largest_over_threshold
            keep_all=False,
        )
        self.FaceVerification = InceptionResnetV1(
            classify=False, pretrained="vggface2"
        ).cuda()
        self.FaceVerification.eval()

    def detect_faces(self, imgs1, imgs2):
        IMG1 = []
        IMG2 = []
        for idx in range(min(imgs1.shape[0], imgs2.shape[0])):
            imgs1_o = imgs1[idx]
            imgs2_o = imgs2[idx]

            img1_cropped = self.mtcnn(imgs1_o)
            img2_cropped = self.mtcnn(imgs2_o)
            if img1_cropped == None:
                temp = torch.ones((3, 160, 160))
                IMG1.append(temp)
            else:
                IMG1.append(img1_cropped)

            if img2_cropped == None:
                temp = torch.ones((3, 160, 160))
                IMG2.append(temp)
            else:
                IMG2.append(img2_cropped)

        IMG1 = torch.stack(IMG1, dim=0).cuda()
        IMG2 = torch.stack(IMG2, dim=0).cuda()
        return IMG1, IMG2

    def compare(self, imgs1, imgs2):
        count = 0
        img1_cropped, img2_cropped = self.detect_faces(imgs1, imgs2)

        with torch.no_grad():
            img1_embeddings = self.FaceVerification(img1_cropped).detach().cpu()
            img2_embeddings = self.FaceVerification(img2_cropped).detach().cpu()

            dists = [
                (e1 - e2).norm().item()
                for e1, e2 in zip(img1_embeddings, img2_embeddings)
            ]

            for dist in dists:
                if dist < self.threshold:
                    count += 1

        return count / img1_cropped.shape[0], sum(dists) / len(dists)

    def count_matching_imgs(self, imgs1, imgs2):
        count = 0
        img1_cropped, img2_cropped = self.detect_faces(imgs1, imgs2)
        with torch.no_grad():
            img1_embeddings = self.FaceVerification(img1_cropped).detach().cpu()
            img2_embeddings = self.FaceVerification(img2_cropped).detach().cpu()

            dists = [
                (e1 - e2).norm().item()
                for e1, e2 in zip(img1_embeddings, img2_embeddings)
            ]

            count += sum(1 for dist in dists if dist < self.threshold)

        return count / img1_cropped.shape[0]

    def get_image_distance(self, img1: np.ndarray, img2: np.ndarray):
        img1_cropped = self.mtcnn(img1)
        img2_cropped = self.mtcnn(img2)

        if img1_cropped is None or img2_cropped is None:
            return math.nan

        img1_embeddings = self.FaceVerification(img1_cropped.unsqueeze(0).cuda())
        img2_embeddings = self.FaceVerification(img2_cropped.unsqueeze(0).cuda())

        with torch.no_grad():
            distance = (img1_embeddings - img2_embeddings).norm().item()

        return distance

    def get_images_distance(
        self, imgs1: torch.tensor, imgs2: torch.tensor
    ) -> list[float]:
        distances = []
        if imgs1.shape != imgs2.shape:
            return distances

        imgs1_ndarray = imgs1.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255.0
        imgs2_ndarray = imgs2.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255.0

        for i in range(imgs1_ndarray.shape[0]):
            try:
                img1_cropped = self.mtcnn(imgs1_ndarray[i]).unsqueeze(0).cuda()
                img2_cropped = self.mtcnn(imgs2_ndarray[i]).unsqueeze(0).cuda()

                img1_embeddings = self.FaceVerification(img1_cropped).detach().cpu()
                img2_embeddings = self.FaceVerification(img2_cropped).detach().cpu()

                distances.append((img1_embeddings - img2_embeddings).norm().item())
            except Exception as e:
                distances.append(math.nan)

        return distances

    def __get_face_recognition(
        self, logger, img1: tensor, img2: tensor
    ) -> tuple[int, int]:
        import requests
        import base64
        from PIL import Image
        from io import BytesIO
        import time

        buffered1 = BytesIO()
        img1 = img1 * 255
        img_image = Image.fromarray(img1.cpu().permute(1, 2, 0).byte().numpy())
        img_image.save(buffered1, format="PNG")
        img1_base64 = base64.b64encode(buffered1.getvalue()).decode("utf-8")

        buffered2 = BytesIO()
        img2 = img2 * 255
        img_image = Image.fromarray(img2.cpu().permute(1, 2, 0).byte().numpy())
        img_image.save(buffered2, format="PNG")
        img2_base64 = base64.b64encode(buffered2.getvalue()).decode("utf-8")

        url = "https://api-us.faceplusplus.com/facepp/v3/compare"
        payload = {
            "api_key": "4grmq8PU5uq4TZMLDrQG-daNc_AMbQhf",
            "api_secret": "qA4JNTxBIEn3NqXPLDBvRcrdhs1yLYaI",
            "image_base64_1": img1_base64,
            "image_base64_2": img2_base64,
        }

        fail_count = 0
        while fail_count < 5:
            response = requests.post(url, data=payload)
            if response.status_code == 200:
                response = response.json()
                return (
                    (1, 1)
                    if response["confidence"] > response["thresholds"]["1e-5"]
                    else (0, 1)
                )
            elif response.status_code == 403:
                fail_count += 1
                time.sleep(0.5)
            else:
                logger.error(response.status_code)
                return (0, 0)

    def get_face_effectiveness(
        self,
        logger,
        source_imgs: tensor,
        swap_imgs: tensor,
        anchor_imgs: tensor,
        pert_swap_imgs: tensor,
    ) -> dict:
        effectivenesses = {"swap": (0, 0), "pert_swap": (0, 0)}
        for i in range(source_imgs.shape[0]):
            source_swap_result = self.__get_face_recognition(
                logger, source_imgs[i], pert_swap_imgs[i]
            )
            effectivenesses["swap"] = tuple(
                a + b for a, b in zip(effectivenesses["swap"], source_swap_result)
            )

            anchor_swap_result = self.__get_face_recognition(
                logger, anchor_imgs[i], pert_swap_imgs[i]
            )
            effectivenesses["pert_swap"] = tuple(
                a + b for a, b in zip(effectivenesses["pert_swap"], anchor_swap_result)
            )

        return effectivenesses

    def calculate_effectiveness(
        self,
        source_imgs: tensor,
        pert_imgs: tensor,
        swap_imgs: tensor,
        pert_swap_imgs: tensor,
        anchor_imgs: tensor,
    ) -> dict:
        effectivenesses = {"pert": 0, "swap": 0, "pert_swap": 0, "anchor": 0}

        source_imgs_ndarray = (
            source_imgs.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255.0
        )
        pert_imgs_ndarray = (
            pert_imgs.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255.0
        )
        swap_imgs_ndarray = (
            swap_imgs.detach().cpu().numpy().transpose(0, 2, 3, 1)
        ) * 255.0
        pert_swap_imgs_ndarray = (
            pert_swap_imgs.detach().cpu().numpy().transpose(0, 2, 3, 1)
        ) * 255.0
        anchor_imgs_ndarray = (
            anchor_imgs.detach().cpu().numpy().transpose(0, 2, 3, 1)
        ) * 255.0

        pert = self.count_matching_imgs(source_imgs_ndarray, pert_imgs_ndarray)
        effectivenesses["pert"] = pert
        swap = self.count_matching_imgs(source_imgs_ndarray, swap_imgs_ndarray)
        effectivenesses["swap"] = swap
        pert_swap = self.count_matching_imgs(
            source_imgs_ndarray, pert_swap_imgs_ndarray
        )
        effectivenesses["pert_swap"] = pert_swap
        anchor = self.count_matching_imgs(anchor_imgs_ndarray, pert_swap_imgs_ndarray)
        effectivenesses["anchor"] = anchor

        return effectivenesses
