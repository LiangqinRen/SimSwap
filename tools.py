import ast
import os


def analyze_efficiency_difference():
    diffs = []
    with open(os.path.join("data", "difference.txt"), "r") as file:
        for line in file:
            tuple_data = ast.literal_eval(line.strip())
            diffs.append(tuple_data[1])

    sort_diffs = sorted(diffs)
    for i in range(1, 100, 1):
        print(f"{i:3}: {sort_diffs[int(len(sort_diffs)/100*i)]:.5f}")


def main():
    analyze_efficiency_difference()


if __name__ == "__main__":
    main()
