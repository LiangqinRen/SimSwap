import cv2
import os
import torch
import numpy as np

from facenet_pytorch import MTCNN, InceptionResnetV1
from skimage import metrics


class Utility:
    def __init__(self):
        pass

    def compare(self, imgs1_list, imgs2_list):
        MSE = []
        SSIM = []
        PSNR = []
        for idx in range(min(imgs1_list.shape[0], imgs2_list.shape[0])):
            img1 = imgs1_list[idx]
            img2 = imgs2_list[idx]
            mse = metrics.mean_squared_error(img1, img2)
            ssim = metrics.structural_similarity(img1, img2, channel_axis=2)
            psnr = metrics.peak_signal_noise_ratio(img1, img2, data_range=1)
            MSE.append(mse)
            SSIM.append(ssim)
            PSNR.append(psnr)

        return np.mean(MSE), np.mean(SSIM), np.mean(PSNR)


class Efficiency:
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
        img1_cropped, img2_cropped = self.detect_face(
            imgs1, imgs2
        )  # 0~1, bs x 3 x 160 x 160

        with torch.no_grad():
            img1_embeddings = self.FaceVerification(img1_cropped).detach().cpu()
            img2_embeddings = self.FaceVerification(img2_cropped).detach().cpu()

            dists = [
                (e1 - e2).norm().item()
                for e1, e2 in zip(img1_embeddings, img2_embeddings)
            ]

            for dist in dists:
                if dist < 0.58:
                    count += 1

        return count / img1_cropped.shape[0]

    def detect_face(self, imgs1, imgs2):
        IMG1 = []
        IMG2 = []
        for idx in range(min(imgs1.shape[0], imgs2.shape[0])):
            imgs1_o = imgs1[idx : idx + 1]
            imgs2_o = imgs2[idx : idx + 1]

            img1_cropped = self.mtcnn(imgs1_o, save_path=None)[0]
            img2_cropped = self.mtcnn(imgs2_o, save_path=None)[0]

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


class Evaluate:
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger

        self.utility = Utility()
        self.efficiency = Efficiency(None)

    def _load_images(self, dir: str):
        imgs_dir = os.path.join(self.args.data_dir, dir)
        imgs_path = [os.path.join(imgs_dir, img) for img in os.listdir(imgs_dir)]

        iter_all_images = (cv2.imread(fn) for fn in imgs_path)
        for i, image in enumerate(iter_all_images):
            if i == 0:
                all_images = np.empty(
                    (len(imgs_path),) + image.shape, dtype=image.dtype
                )
            all_images[i] = image
        return all_images

    def evaluate(self):
        A_imgs = self._load_images(self.args.eval_A)
        B_imgs = self._load_images(self.args.eval_B)

        efficiency = self.efficiency.compare(A_imgs, B_imgs)
        mse, ssim, psnr = self.utility.compare(A_imgs, B_imgs)
        self.logger.info(
            f"Compare {self.args.eval_A} with {self.args.eval_B}, efficiency: {efficiency:.3f}, utility: {mse:.3f}, {ssim:.3f}, {psnr:.3f}"
        )
