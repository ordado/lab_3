import random
from cluster import Cluster


class KMeans:
    def __init__(self, k, num_iterations):
        # properties and functions starting with _ are invisible outside of class
        # very useful to ease the usage of class
        self._k = k
        self._num_iterations = num_iterations
        self._clusters = []

    def run(self, points, random_seed):
        random.seed(random_seed)
        # Randomly initiate clusters
        self._clusters = []
        initial_centroids = random.sample(points, self._k)
        for i, initial_centroid in enumerate(initial_centroids):
            new_cluster = Cluster(i, initial_centroid)
            self._clusters.append(new_cluster)

        for current_iteration in range(self._num_iterations):
            # Clear all clusters
            for cluster in self._clusters:
                cluster.remove_point()

            # Re-assign all points
            for point in points:
                distances_to_clusters = {x.id: point.distance_to(x.centroid) for x in self._clusters}
                closest_cluster_id = sorted(distances_to_clusters.keys(), key=lambda x: distances_to_clusters[x])[0]
                self._clusters[closest_cluster_id].add_point(point)

            # Recompute centroids and look if change happened
            changes = [cluster.compute_centroid() for cluster in self._clusters]
            if sum(changes) == 0:  # if everyone is False then sum is 0
                break

    def print_results(self):
        total_loss = 0
        for cluster in self._clusters:
            total_loss += cluster.compute_loss()
        print('{:.1f}'.format(total_loss))
