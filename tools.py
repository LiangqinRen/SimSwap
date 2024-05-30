import ast
import os
import numpy as np


def analyze_efficiency_difference():
    diffs = []
    with open(os.path.join("data", "difference.txt"), "r") as file:
        for line in file:
            tuple_data = ast.literal_eval(line.strip())
            diffs.append(tuple_data[1])

    sort_diffs = sorted(diffs)
    for i in range(1, 100, 1):
        print(f"{i:3}: {sort_diffs[int(len(sort_diffs)/100*i)]:.5f}")


def calculate_score(
    utility: float, efficiency_clean: float, efficiency_pert: float
) -> float:
    print(
        f"score: {utility:.3f},{efficiency_clean:.3f},{efficiency_pert:.3f} -> {pow(utility, 5) + np.tanh(3*(efficiency_clean - efficiency_pert)):.3f}"
    )


def main():
    calculate_score(0.830, 0.741, 0.007)
    calculate_score(0.699, 0.741, 0.633)


if __name__ == "__main__":
    main()
