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
        train_set_path = join(self.args.data_dir, "train")
        if version == 1:
            imgs_path = []
            for people in all_people:
                people_path = join(train_set_path, people)
                all_image = sorted(os.listdir(people_path))
                selected_imgs_name = random.sample(
                    all_image, min(self.args.metric_people_image, len(all_image))
                )
                imgs_path.extend(
                    [join(people_path, name) for name in selected_imgs_name]
                )
            return imgs_path, imgs_path
        elif version == 2:
            source_imgs_path, target_imgs_path = [], []
            for i, people in enumerate(all_people):
                people_path = join(train_set_path, people)
                all_image = sorted(os.listdir(people_path))
                selected_imgs_name = random.sample(
                    all_image, min(self.args.metric_people_image, len(all_image))
                )

                if i % 2 == 0:
                    source_imgs_path.extend(
                        [join(people_path, name) for name in selected_imgs_name]
                    )
                else:
                    target_imgs_path.extend(
                        [join(people_path, name) for name in selected_imgs_name]
                    )
            return source_imgs_path, target_imgs_path
        elif version == 3:
            source_imgs_path, target_imgs_path = [], []
            for people in all_people:
                people_path = join(train_set_path, people)
                all_image = sorted(os.listdir(people_path))
                selected_imgs_name = random.sample(
                    all_image, min(self.args.metric_people_image * 2, len(all_image))
                )
                if len(selected_imgs_name) % 2 == 1:
                    selected_imgs_name.pop()

                for i, image in enumerate(selected_imgs_name):
                    if i % 2 == 0:
                        source_imgs_path.append(join(people_path, image))
                    else:
                        target_imgs_path.append(join(people_path, image))
            return source_imgs_path, target_imgs_path

    def calculate_effectiveness_threshold(self, version: int) -> None:
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name} version {version}",
        )
        train_set_path = join(self.args.data_dir, "train")
        all_people = sorted(os.listdir(train_set_path))

        source_imgs_path, target_imgs_path = self.__get_paired_images_path(
            all_people, version
        )

        path_distances = []
        sum_difference = 0
        total_batch = (
            min(len(source_imgs_path), len(target_imgs_path)) // self.args.batch_size
        )
        for i in tqdm(range(total_batch)):
            iter_source_path = source_imgs_path[
                i * self.args.batch_size : (i + 1) * self.args.batch_size
            ]
            iter_target_path = target_imgs_path[
                i * self.args.batch_size : (i + 1) * self.args.batch_size
            ]

            source_imgs = super()._load_imgs(iter_source_path)
            target_imgs = super()._load_imgs(iter_target_path)

            if version in [1, 2]:
                source_identity = super()._get_imgs_identity(source_imgs)
                swap_imgs = self.target(None, target_imgs, source_identity, None, True)
                distances = self.effectiveness.get_image_distance(
                    source_imgs, swap_imgs
                )
            else:
                distances = self.effectiveness.get_image_distance(
                    source_imgs, target_imgs
                )
            for i in range(len(distances)):
                if distances[i] == math.nan:
                    continue
                tqdm.write(
                    f"{iter_source_path[i]}, {iter_target_path[i]} distance: {distances[i]:.5f}"
                )
                sum_difference += distances[i]
                path_distances.append(
                    (iter_source_path[i], iter_target_path[i], distances[i])
                )

        sorted_distances = sorted(path_distances, key=lambda x: x[2])

        with open(join(self.args.log_dir, f"v{version}_distance.txt"), "w") as f:
            for line in sorted_distances:
                f.write(f"{line}\n")

        self.logger.info(
            f"With {len(distances)} pictures, the max, mean, min distances are {sorted_distances[-1][2]:.5f}, {sum_difference/len(distances):.5f} and {sorted_distances[0][2]:.5f}"
        )


def main(args, logger):
    worker = Worker(args, logger)
    worker.calculate_effectiveness_threshold(version=1)
    worker.calculate_effectiveness_threshold(version=2)
    worker.calculate_effectiveness_threshold(version=3)


if __name__ == "__main__":
    main()
