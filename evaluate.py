import torch
import lpips
import numpy as np
import math
import requests
import base64
import time

from facenet_pytorch import MTCNN, InceptionResnetV1
from skimage import metrics
from torch import tensor
import face_recognition
from PIL import Image
from io import BytesIO


class Utility:
    lpips_distance = lpips.LPIPS(net="vgg", verbose=False).cuda()

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger

    def calculate_utility(self, imgs1: torch.tensor, imgs2: torch.tensor):
        utilities = {"mse": [], "psnr": [], "ssim": [], "lpips": []}

        imgs1_ndarray = imgs1.detach().cpu().numpy().transpose(0, 2, 3, 1)
        imgs2_ndarray = imgs2.detach().cpu().numpy().transpose(0, 2, 3, 1)
        for i in range(min(imgs1.shape[0], imgs2.shape[0])):
            mse = metrics.mean_squared_error(imgs1_ndarray[i], imgs2_ndarray[i])
            utilities["mse"].append(mse)

            psnr = metrics.peak_signal_noise_ratio(
                imgs1_ndarray[i], imgs2_ndarray[i], data_range=1
            )
            utilities["psnr"].append(psnr)

            ssim = metrics.structural_similarity(
                imgs1_ndarray[i],
                imgs2_ndarray[i],
                channel_axis=2,
                data_range=1,
            )
            utilities["ssim"].append(ssim)

            lpips_score = self.lpips_distance(imgs1[i], imgs2[i])
            utilities["lpips"].append(lpips_score.detach().cpu().numpy())

        for i in utilities:
            utilities[i] = np.mean(utilities[i])

        return utilities


class Effectiveness:
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger

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

    def __get_facerec_matching(self, imgs1, imgs2):
        matching_count, valid_count = 0, 0
        for i in range(imgs1.shape[0]):
            try:
                img1, img2 = imgs1[i], imgs2[i]
                img1 = np.array(
                    Image.fromarray(
                        (img1.detach().cpu().permute(1, 2, 0).numpy() * 255).astype(
                            np.uint8
                        )
                    )
                )
                img2 = np.array(
                    Image.fromarray(
                        (img2.detach().cpu().permute(1, 2, 0).numpy() * 255).astype(
                            np.uint8
                        )
                    )
                )
                img1_encoding = face_recognition.face_encodings(img1)[0]
                img2_encoding = face_recognition.face_encodings(img2)[0]
                face_distances = face_recognition.face_distance(
                    [img1_encoding], img2_encoding
                )
                i, face_distance = next(enumerate(face_distances))
                if face_distance <= 0.6:
                    matching_count += 1
                valid_count += 1
            except IndexError:
                valid_count += 1
            except Exception as e:
                self.logger.warning(e)

        return (matching_count, valid_count)

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

    def __get_facepp_matching_single(
        self, img1: tensor, img2: tensor, key: str, secret: str
    ):
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
            "api_key": key,
            "api_secret": secret,
            "image_base64_1": img1_base64,
            "image_base64_2": img2_base64,
        }

        fail_count = 0
        while fail_count < 10:
            try:
                response = requests.post(url, data=payload)
                if response.status_code == 200:
                    response = response.json()
                    if "confidence" in response:
                        return (
                            (1, 1)
                            if response["confidence"] > response["thresholds"]["1e-5"]
                            else (0, 1)
                        )
                    elif ("faces1" in response and len(response["faces1"]) == 0) or (
                        "faces2" in response and len(response["faces2"]) == 0
                    ):
                        return (0, 1)
                    else:
                        self.logger.warning(response)
                        return (0, 1e-10)
                elif response.status_code == 400:
                    return (0, 1)
                elif response.status_code == 403:
                    time.sleep(0.3)
                    fail_count += 1
                else:
                    self.logger.error(response)
                    return (0, 1e-10)
            except BaseException as e:
                self.logger.error(e)
                return (0, 1e-10)

        return (0, 1e-10)

    def __get_facepp_matching(self, imgs1: tensor, imgs2: tensor):
        from concurrent.futures import ThreadPoolExecutor

        assert imgs1.shape == imgs2.shape

        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    self.__get_facepp_matching_single,
                    imgs1[i],
                    imgs2[i],
                    self.args.facepp_api_key[i % len(self.args.facepp_api_key)],
                    self.args.facepp_api_secret[i % len(self.args.facepp_api_secret)],
                )
                for i in range(imgs1.shape[0])
            ]
            results = [future.result() for future in futures]

        success_count, total_count = 0, 0
        for result in results:
            success_count += result[0]
            total_count += result[1]

        return (success_count, total_count)

    def calculate_single_effectiveness(self, imgs1: tensor, imgs2: tensor) -> dict:
        effectivenesses = {}

        effectivenesses["facerec"] = self.__get_facerec_matching(imgs1, imgs2)
        effectivenesses["face++"] = self.__get_facepp_matching(imgs1, imgs2)

        return effectivenesses

    def calculate_effectiveness(
        self,
        source_imgs: tensor,
        pert_imgs: tensor,
        swap_imgs: tensor,
        pert_swap_imgs: tensor,
        anchor_imgs: tensor,
    ) -> dict:
        effectivenesses = {"facerec": {}, "face++": {}}
        if pert_imgs is not None:
            effectivenesses["facerec"]["pert"] = self.__get_facerec_matching(
                source_imgs, pert_imgs
            )
            effectivenesses["face++"]["pert"] = self.__get_facepp_matching(
                source_imgs, pert_imgs
            )

        effectivenesses["facerec"]["swap"] = self.__get_facerec_matching(
            source_imgs, swap_imgs
        )
        effectivenesses["face++"]["swap"] = self.__get_facepp_matching(
            source_imgs, swap_imgs
        )

        effectivenesses["facerec"]["pert_swap"] = self.__get_facerec_matching(
            source_imgs, pert_swap_imgs
        )
        effectivenesses["face++"]["pert_swap"] = self.__get_facepp_matching(
            source_imgs, pert_swap_imgs
        )

        if anchor_imgs is not None:
            effectivenesses["facerec"]["anchor"] = self.__get_facerec_matching(
                pert_swap_imgs, anchor_imgs
            )
            effectivenesses["face++"]["anchor"] = self.__get_facepp_matching(
                pert_swap_imgs, anchor_imgs
            )

        return effectivenesses
