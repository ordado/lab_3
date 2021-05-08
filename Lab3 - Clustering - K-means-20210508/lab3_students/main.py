from sys import argv
import os
from point import Point
from k_means import KMeans


def load_data(input_path):
    """
    Loads data from given csv
    :param input_path: path to csv file
    :return: returns data as list of Point
    """
    points = []
    with open(input_path, 'r', encoding='utf8') as f:
        for row in f.readlines():
            row = row.strip()
            values = row.split(' ')
            points.append(Point(values[0], values[1:]))
    return points


def run_kmeans():
    list_seed = [1, 1, 1, 12, 12, 12]
    list_k = [3, 4, 5, 3, 4, 5]

    print("seed k sl1")

    for index, value_seed in enumerate(list_seed):
        k = list_k[index]
        num_iterations = 10
        input_path = "colors_dataset_ready.txt"
        random_seed = value_seed

        if k <= 1 or num_iterations <= 0:
            print('Please provide correct parameters')
            exit(1)
        if not os.path.exists(input_path):
            print('Input file does not exist')
            exit(1)

        points = load_data(input_path)
        if k >= len(points):
            print('Please set K less than size of dataset')
            exit(1)

        runner = KMeans(k, num_iterations)
        runner.run(points, random_seed)
        print(list_seed[index], end=" ")
        print(list_k[index], end=" ")
        runner.print_results()


if __name__ == '__main__':
    run_kmeans()
