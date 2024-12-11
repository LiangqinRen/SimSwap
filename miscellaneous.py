import common_base

import os
import random
import shutil
import math
import inspect

from tqdm import tqdm
from os.path import join


class Worker(common_base.Base):
    def __init__(self, args, logger):
        super().__init__(args, logger)

    def split_dataset(self) -> None:
        all_people = sorted(os.listdir(self.dataset_dir))
        testset_people = random.sample(
            all_people, int(len(all_people) * self.args.testset_percentage)
        )

        os.makedirs(join(self.args.data_dir, "train"), exist_ok=True)
        os.makedirs(join(self.args.data_dir, "test"), exist_ok=True)

        for i, people in enumerate(all_people, start=1):
            self.logger.info(f"{i:4}/{len(all_people):4}|Copy folder {people}")
            shutil.copytree(
                join(self.dataset_dir, people),
                (
                    join(self.args.data_dir, "test", people)
                    if people in testset_people
                    else join(self.args.data_dir, "train", people)
                ),
            )

    def __get_paired_images_path(
        self, all_people: list[str], version: int
    ) -> tuple[list[str], list[str]]:
        test_set_path = join(self.args.data_dir, "test")
        imgs1_path, imgs2_path = [], []
        if version == 1:
            # distance between same people's image
            for i, people in enumerate(all_people):
                people_path = join(test_set_path, people)
                people_imgs = sorted(os.listdir(people_path))

                if len(people_imgs) % 2 == 1:
                    people_imgs.pop()

                for j, name in enumerate(people_imgs):
                    if j % 2 == 0:
                        imgs1_path.append(join(people_path, name))
                    else:
                        imgs2_path.append(join(people_path, name))
            assert len(imgs1_path) == len(imgs2_path)
        elif version == 2:
            # distance between different people's image
            for i, people in enumerate(all_people):
                people_path = join(test_set_path, people)
                people_imgs = sorted(os.listdir(people_path))

                if i % 2 == 0:
                    imgs1_path.extend([join(people_path, name) for name in people_imgs])
                else:
                    imgs2_path.extend([join(people_path, name) for name in people_imgs])

            random.shuffle(imgs1_path)
            random.shuffle(imgs2_path)

        return imgs1_path, imgs2_path

    def calculate_effectiveness_distance(self, version: int) -> None:
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name} version {version}",
        )
        test_set_path = join(self.args.data_dir, "test")
        all_people = sorted(os.listdir(test_set_path))

        imgs1_path, imgs2_path = self.__get_paired_images_path(all_people, version)
        path_distances = []
        distance_sum = 0
        total_batch = min(len(imgs1_path), len(imgs2_path)) // self.args.batch_size
        for i in tqdm(range(total_batch)):
            iter_imgs1_path = imgs1_path[
                i * self.args.batch_size : (i + 1) * self.args.batch_size
            ]
            iter_imgs2_path = imgs2_path[
                i * self.args.batch_size : (i + 1) * self.args.batch_size
            ]

            imgs1 = self._load_imgs(iter_imgs1_path)
            imgs2 = self._load_imgs(iter_imgs2_path)

            distances = self.effectiveness.get_images_distance(imgs1, imgs2)
            for i in range(len(distances)):
                if distances[i] is math.nan:
                    continue
                tqdm.write(
                    f"{iter_imgs1_path[i]}, {iter_imgs2_path[i]} distance: {distances[i]:.5f}"
                )
                distance_sum += distances[i]
                path_distances.append(
                    (iter_imgs1_path[i], iter_imgs2_path[i], distances[i])
                )

        sorted_path_distances = sorted(path_distances, key=lambda x: x[2])

        with open(join(self.args.log_dir, f"v{version}_distance.txt"), "w") as f:
            for line in sorted_path_distances:
                f.write(f"{line}\n")

        self.logger.info(
            f"With {len(sorted_path_distances)} pictures, the max, mean, min distances are {sorted_path_distances[-1][2]:.5f}, {distance_sum/len(sorted_path_distances):.5f} and {sorted_path_distances[0][2]:.5f}"
        )


def main(args, logger):
    worker = Worker(args, logger)
    worker.calculate_effectiveness_distance(version=1)
    worker.calculate_effectiveness_distance(version=2)


if __name__ == "__main__":
    main()
