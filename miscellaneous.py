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

    def calculate_effectiveness_threshold_v1(self) -> None:
        # swap with himself, calculate the distance, sort(small->large), select 99%
        train_set_path = join(self.args.data_dir, "train")
        all_people = sorted(os.listdir(train_set_path))

        imgs_path = []
        for people in all_people:
            people_path = join(train_set_path, people)
            all_image = sorted(os.listdir(people_path))
            selected_imgs_name = random.sample(
                all_image, min(self.args.metric_people_image, len(all_image))
            )
            imgs_path.extend([join(people_path, name) for name in selected_imgs_name])

        distances = []
        sum_difference = 0
        for path in tqdm(imgs_path):
            source_img = super()._load_imgs([path])
            source_identity = super()._get_imgs_identity(source_img)
            target_img = super()._load_imgs([path])
            swap_img = self.target(None, target_img, source_identity, None, True)

            distance = self.effectiveness.get_image_distance(source_img, swap_img)
            if math.isinf(distance):
                continue

            sum_difference += distance

            distances.append((path, distance))
            tqdm.write(f"{path} distance: {distance:.5f}")

        sorted_distances = sorted(distances, key=lambda x: x[1])

        with open(join(self.args.log_dir, "distance.txt"), "w") as f:
            for line in sorted_distances:
                f.write(f"{line}\n")

        self.logger.info(
            f"With {len(distances)} pictures, the max, mean, min distances are {sorted_distances[-1][1]:.5f}, {sum_difference/len(distances):.5f} and {sorted_distances[0][1]:.5f}"
        )


def main(args, logger):
    worker = Worker(args, logger)
    worker.calculate_effectiveness_threshold_v1()


if __name__ == "__main__":
    main()
