import cv2
import os
import torch
import lpips
import numpy as np

from facenet_pytorch import MTCNN, InceptionResnetV1
from skimage import metrics
from torchvision.models import vgg16, VGG16_Weights


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
        utilities = {"mse": [], "ssim": [], "psnr": [], "lpips": []}

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
    def __init__(self, opt):
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

    def compare(self, imgs1, imgs2):
        count = 0
        img1_cropped, img2_cropped = self.detect_face(imgs1, imgs2)

        with torch.no_grad():
            img1_embeddings = self.FaceVerification(img1_cropped).detach().cpu()
            img2_embeddings = self.FaceVerification(img2_cropped).detach().cpu()

            dists = [
                (e1 - e2).norm().item()
                for e1, e2 in zip(img1_embeddings, img2_embeddings)
            ]

            for dist in dists:
                if dist < 0.91906:
                    count += 1

        return count / img1_cropped.shape[0], sum(dists) / len(dists)

    def detect_face(self, imgs1, imgs2):
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

    def _is_ndarray_valid(self, ndarray: np.ndarray) -> bool:
        return ndarray is not None and isinstance(ndarray, np.ndarray)

    def _is_tensor_valid(self, tensor: torch.tensor) -> bool:
        return tensor is not None and isinstance(tensor, torch.Tensor)

    def get_image_difference(self, source: torch.tensor, swap: torch.tensor) -> float:
        try:
            source_ndarray = source.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255.0
            swap_ndarray = swap.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255.0

            source_cropped = self.mtcnn(source_ndarray[0])[None, :].cuda()
            swap_cropped = self.mtcnn(swap_ndarray[0])[None, :].cuda()

            source_embeddings = self.FaceVerification(source_cropped).detach().cpu()
            swap_embeddings = self.FaceVerification(swap_cropped).detach().cpu()

            return (source_embeddings - swap_embeddings).norm().item()
        except:
            return float("inf")
