import torch
import lpips
import math
import requests
import base64
import time
import os
import warnings
import boto3
import random
import pickle
import numpy as np
import torchvision.transforms as transforms

from concurrent.futures import ThreadPoolExecutor
from facenet_pytorch import MTCNN, InceptionResnetV1
from skimage import metrics
from torch import tensor
import face_recognition
from PIL import Image
from io import BytesIO
from os.path import join


class Utility:
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self.lpips_distance = lpips.LPIPS(net="vgg", verbose=False).cuda()

    def calculate_utility(self, imgs1: torch.tensor, imgs2: torch.tensor):
        utilities = {"mse": [], "psnr": [], "ssim": [], "lpips": []}

        imgs1_ndarray = imgs1.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255.0
        imgs2_ndarray = imgs2.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255.0
        for i in range(min(imgs1.shape[0], imgs2.shape[0])):
            mse = metrics.mean_squared_error(imgs1_ndarray[i], imgs2_ndarray[i])
            utilities["mse"].append(mse)

            utilities["psnr"].append(
                metrics.peak_signal_noise_ratio(
                    imgs1_ndarray[i], imgs2_ndarray[i], data_range=255
                )
            )

            utilities["ssim"].append(
                metrics.structural_similarity(
                    imgs1_ndarray[i],
                    imgs2_ndarray[i],
                    channel_axis=2,
                    data_range=255,
                )
            )

            lpips_score = self.lpips_distance(imgs1[i], imgs2[i])
            utilities["lpips"].append(lpips_score.detach().cpu().numpy())

        for i in utilities:
            utilities[i] = np.mean(utilities[i])

        return utilities


class Effectiveness:
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger

        self.candi_funcs = self.__init_functions()

        self.mtcnn = MTCNN(
            image_size=160,
            device="cuda",
            selection_method="largest",
            keep_all=False,
        )
        self.FaceVerification = InceptionResnetV1(
            classify=False, pretrained="vggface2"
        ).cuda()
        self.FaceVerification.eval()

        self.aws_client = boto3.client(
            "rekognition",
            aws_access_key_id=self.args.effectiveness["aws"]["api_key"],
            aws_secret_access_key=self.args.effectiveness["aws"]["api_secret"],
            region_name=self.args.effectiveness["aws"]["api_region"],
        )

    def __init_functions(self) -> dict:
        candi_funcs = {
            "facenet": self.__get_facenet_matching,
            "facerec": self.__get_facerec_matching,
            "face++": self.__get_facepp_matching,
            "aws": self.__get_aws_matching,
        }
        for k, v in self.args.effectiveness.items():
            if v["use"] is False:
                del candi_funcs[k]

        return candi_funcs

    def __get_facenet_matching(self, imgs1, imgs2):
        matching_count, valid_count = 0, 1e-10
        distances = self.get_images_distance(imgs1, imgs2)
        for distance in distances:
            if distance is math.nan:
                continue
            else:
                matching_count += (
                    distance <= self.args.effectiveness["facenet"]["threshold"]
                )
                valid_count += 1

        return (matching_count, valid_count)

    def __get_facerec_matching(self, imgs1, imgs2):
        matching_count, valid_count = 0, 1e-10
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
                img1_encoding = face_recognition.face_encodings(img1, model="large")[0]
                img2_encoding = face_recognition.face_encodings(img2, model="large")[0]
                matching_count += face_recognition.compare_faces(
                    [img1_encoding], img2_encoding
                )[0]
                valid_count += 1
            except IndexError:
                valid_count += 1
            except Exception as e:
                self.logger.warning(e)

        return (matching_count, valid_count)

    def __get_aws_matching(self, imgs1, imgs2) -> tuple[int, int]:
        matching_count, valid_count = 0, 1e-10
        for img1, img2 in zip(imgs1, imgs2):
            try:
                img1 = Image.fromarray(
                    (img1.detach().cpu().permute(1, 2, 0).numpy() * 255)
                    .clip(0, 255)
                    .astype(np.uint8)
                )
                buffer1 = BytesIO()
                img1.save(buffer1, format="png")
                img_bytes1 = buffer1.getvalue()

                img2 = Image.fromarray(
                    (img2.detach().cpu().permute(1, 2, 0).numpy() * 255)
                    .clip(0, 255)
                    .astype(np.uint8)
                )
                buffer2 = BytesIO()
                img2.save(buffer2, format="png")
                img_bytes2 = buffer2.getvalue()

                response = self.aws_client.compare_faces(
                    SimilarityThreshold=80,
                    SourceImage={"Bytes": img_bytes1},
                    TargetImage={"Bytes": img_bytes2},
                )

                matching_count += len(response["FaceMatches"])
                valid_count += 1
            except Exception as e:
                error_code = e.response["Error"]["Code"]
                if error_code == "InvalidParameterException":
                    valid_count += 1
                else:
                    self.logger.error(e)
                    valid_count += 1e-10

        return (matching_count, valid_count)

    def __get_facepp_matching_single(
        self, img1: tensor, img2: tensor, key: str, secret: str
    ):
        buffer1 = BytesIO()
        img1 = img1 * 255
        img_image = Image.fromarray(img1.cpu().permute(1, 2, 0).byte().numpy())
        img_image.save(buffer1, format="PNG")
        img1_base64 = base64.b64encode(buffer1.getvalue()).decode("utf-8")

        buffer2 = BytesIO()
        img2 = img2 * 255
        img_image = Image.fromarray(img2.cpu().permute(1, 2, 0).byte().numpy())
        img_image.save(buffer2, format="PNG")
        img2_base64 = base64.b64encode(buffer2.getvalue()).decode("utf-8")

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

        api_keys = self.args.effectiveness["face++"]["api_key"]
        api_secrets = self.args.effectiveness["face++"]["api_secret"]

        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    self.__get_facepp_matching_single,
                    imgs1[i],
                    imgs2[i],
                    api_keys[i % len(api_keys)],
                    api_secrets[i % len(api_secrets)],
                )
                for i in range(imgs1.shape[0])
            ]
            results = [future.result() for future in futures]

        success_count, total_count = 0, 1e-10
        for result in results:
            success_count += result[0]
            total_count += result[1]

        return (success_count, total_count)

    def is_same_identity_via_facepp(self, imgs1: tensor, imgs2: tensor) -> list[bool]:
        api_keys = self.args.effectiveness["face++"]["api_key"]
        api_secrets = self.args.effectiveness["face++"]["api_secret"]

        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    self.__get_facepp_matching_single,
                    imgs1[i],
                    imgs2[i],
                    api_keys[i % len(api_keys)],
                    api_secrets[i % len(api_secrets)],
                )
                for i in range(imgs1.shape[0])
            ]
            matchings = [future.result() for future in futures]

        results = []
        for matching in matchings:
            results.append(matching[0] == 1)

        return results

    # def get_image_distance(self, img1: np.ndarray, img2: np.ndarray):
    #     img1_cropped = self.mtcnn(img1)
    #     img2_cropped = self.mtcnn(img2)

    #     if img1_cropped is None or img2_cropped is None:
    #         return math.nan

    #     img1_embeddings = self.FaceVerification(img1_cropped.unsqueeze(0).cuda())
    #     img2_embeddings = self.FaceVerification(img2_cropped.unsqueeze(0).cuda())

    #     with torch.no_grad():
    #         distance = (img1_embeddings - img2_embeddings).norm().item()

    #     return distance

    def get_image_distance(self, img1: np.ndarray, embeddings: list) -> list:
        img1_cropped = self.mtcnn(img1)

        if img1_cropped is None:
            return [math.nan] * len(embeddings)

        img1_embedding = self.FaceVerification(img1_cropped.unsqueeze(0).cuda())
        distances = []
        with torch.no_grad():
            for embedding in embeddings:
                distance = (embedding - img1_embedding).norm().item()
                distances.append(distance)

        return distances

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
                img1_cropped = self.mtcnn(imgs1_ndarray[i])
                img2_cropped = self.mtcnn(imgs2_ndarray[i])
                if img1_cropped is None or img2_cropped is None:
                    distances.append(float("inf"))
                    continue

                img1_embeddings = (
                    self.FaceVerification(img1_cropped.unsqueeze(0).cuda())
                    .detach()
                    .cpu()
                )
                img2_embeddings = (
                    self.FaceVerification(img2_cropped.unsqueeze(0).cuda())
                    .detach()
                    .cpu()
                )

                distances.append((img1_embeddings - img2_embeddings).norm().item())
            except Exception as e:
                self.logger.warning(e)
                distances.append(math.nan)

        return distances

    def calculate_single_effectiveness(self, imgs1: tensor, imgs2: tensor) -> dict:
        return self.__get_facepp_matching(imgs1, imgs2)

    def calculate_effectiveness(
        self,
        source_imgs: tensor,
        pert_imgs: tensor,
        swap_imgs: tensor,
        pert_swap_imgs: tensor,
        anchor_imgs: tensor,
    ) -> dict:
        effectivenesses = {}
        for k, v in self.candi_funcs.items():
            effectivenesses[k] = {}
            if source_imgs is not None and pert_imgs is not None:
                effectivenesses[k]["pert"] = v(source_imgs, pert_imgs)

            if source_imgs is not None and swap_imgs is not None:
                effectivenesses[k]["swap"] = v(source_imgs, swap_imgs)

            if source_imgs is not None and pert_swap_imgs is not None:
                effectivenesses[k]["pert_swap"] = v(source_imgs, pert_swap_imgs)

            if pert_swap_imgs is not None and anchor_imgs is not None:
                effectivenesses[k]["anchor"] = v(pert_swap_imgs, anchor_imgs)

        return effectivenesses


class Anchor:
    def __init__(self, args, logger, effectiveness):
        self.args = args
        self.logger = logger
        self.effectiveness = effectiveness

        self.anchorset_dir = join(args.data_dir, "anchor", args.anchor_dir)
        self.anchor_imgs = self.__get_anchor_imgs()
        self.anchor_cache = self.__cache_anchor_imgs()

    def __cache_anchor_imgs(self) -> dict:
        mtcnn = MTCNN(
            image_size=160,
            device="cuda",
            selection_method="largest",
            keep_all=False,
        )
        FaceVerification = InceptionResnetV1(
            classify=False, pretrained="vggface2"
        ).cuda()
        FaceVerification.eval()

        anchor_cache = {}
        for k, v in self.anchor_imgs.items():
            imgs_ndarray = v.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255.0
            embeddings = []
            for i, img in enumerate(imgs_ndarray):
                img_cropped = mtcnn(img)
                if img_cropped is None:
                    self.logger.fatal(f"Cannot detect the face from {i}th {k}")
                embedding = FaceVerification(img_cropped.unsqueeze(0).cuda())
                embeddings.append(embedding)
            anchor_cache[k] = embeddings

        return anchor_cache

    def __hash_tensor(self, img: tensor):
        return hash(tuple(img.view(-1).tolist()))

    def __get_anchor_imgs_path(self) -> dict:
        male_imgs_path = sorted(os.listdir(join(self.anchorset_dir, "male")))
        male_imgs_path = [
            join(self.anchorset_dir, "male", name) for name in male_imgs_path
        ]

        female_imgs_path = sorted(os.listdir(join(self.anchorset_dir, "female")))
        female_imgs_path = [
            join(self.anchorset_dir, "female", name) for name in female_imgs_path
        ]

        return {
            "male": male_imgs_path,
            "female": female_imgs_path,
            "mix": male_imgs_path[:15] + female_imgs_path[:15],
        }

    def __load_imgs(self, imgs_path) -> dict:
        transformer = transforms.Compose([transforms.ToTensor()])
        imgs = [transformer(Image.open(path).convert("RGB")) for path in imgs_path]
        imgs = torch.stack(imgs)

        return imgs.cuda()

    def __get_anchor_imgs(self) -> dict:
        anchor_imgs_path = self.__get_anchor_imgs_path()

        return {
            "male": self.__load_imgs(anchor_imgs_path["male"]),
            "female": self.__load_imgs(anchor_imgs_path["female"]),
            "mix": self.__load_imgs(anchor_imgs_path["mix"]),
        }

    def __check_imgs_gender_single(self, img: tensor, key: str, secret: str) -> dict:
        result = {self.__hash_tensor(img): "fail"}

        buffered = BytesIO()
        img_image = img * 255
        img_image = Image.fromarray(img_image.cpu().permute(1, 2, 0).byte().numpy())
        img_image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        url = "https://api-us.faceplusplus.com/facepp/v3/detect"
        payload = {
            "api_key": key,
            "api_secret": secret,
            "image_base64": img_base64,
            "return_attributes": "gender",
        }

        fail_count = 0
        while fail_count < 10:
            try:
                response = requests.post(url, data=payload)
                if response.status_code == 200:
                    response = response.json()
                    if len(response["faces"]) > 1:
                        return result

                    gender = response["faces"][0]["attributes"]["gender"]["value"]
                    result[self.__hash_tensor(img)] = gender.lower()
                    break
                elif response.status_code == 400:
                    self.logger.info(response["time_used"])
                    return result
                elif response.status_code == 403:
                    fail_count += 0.25
                    time.sleep(0.3)
                else:
                    fail_count += 1
                    self.logger.error(response)
            except BaseException as e:
                fail_count += 1
                self.logger.error(e)

        return result

    def __check_imgs_gender(self, imgs: tensor):
        from concurrent.futures import ThreadPoolExecutor

        api_keys = self.args.effectiveness["face++"]["api_key"]
        api_secrets = self.args.effectiveness["face++"]["api_secret"]
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    self.__check_imgs_gender_single,
                    imgs[i],
                    api_keys[i % len(api_keys)],
                    api_secrets[i % len(api_secrets)],
                )
                for i in range(imgs.shape[0])
            ]
            results = [future.result() for future in futures]

        imgs_gender = {}
        for result in results:
            imgs_gender.update(result)

        return imgs_gender

    def find_best_anchors(self, imgs: tensor) -> tensor:
        if not self.args.anchor_mix:
            imgs_gender = self.__check_imgs_gender(imgs)

        imgs_ndarray = imgs.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255.0
        best_anchors = []
        for i in range(imgs.shape[0]):
            if self.args.anchor_mix:
                candidates = self.anchor_imgs["mix"]
                cache = self.anchor_cache["mix"]
            else:
                candidates = (
                    self.anchor_imgs["female"]
                    if imgs_gender[self.__hash_tensor(imgs[i])] == "male"
                    else self.anchor_imgs["male"]
                )
                cache = (
                    self.anchor_cache["female"]
                    if imgs_gender[self.__hash_tensor(imgs[i])] == "male"
                    else self.anchor_cache["male"]
                )

            # img_to_match = imgs[i].unsqueeze(0)
            # img_to_match = img_to_match.repeat(candidates.shape[0], 1, 1, 1)
            # same_identity = self.effectiveness.is_same_identity_via_facepp(
            #     img_to_match, candidates
            # )
            same_identity = [False] * candidates.shape[0]

            results = self.effectiveness.get_image_distance(imgs_ndarray[i], cache)
            distances = []
            for j, distance in enumerate(results):
                if (
                    distance is math.nan
                    or distance <= self.args.anchor_min_distance
                    or same_identity[j]
                ):
                    continue
                distances.append((distance, j))

            sorted_distances = sorted(distances)
            if len(sorted_distances) > self.args.anchor_index:
                best_anchor_idx = sorted_distances[self.args.anchor_index][1]
                best_anchors.append(candidates[best_anchor_idx])
            else:
                best_anchors.append(candidates[-1])

        return torch.stack(best_anchors, dim=0)
