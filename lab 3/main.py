import random
import copy
import matplotlib.pyplot as plt

def generate_test_sequence(N):
    test_sequence = [(random.uniform(0, 1), random.uniform(0, 1)) for _ in range(N)]
    return test_sequence

def distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

def k_means_clustering(test_sequence, num_clusters):
    centroids = random.sample(test_sequence, num_clusters)
    clusters = [[] for _ in range(num_clusters)]

    while True:
        new_clusters = [[] for _ in range(num_clusters)]
        for point in test_sequence:
            closest_centroid_index = min(range(num_clusters), key=lambda i: distance(point, centroids[i]))
            new_clusters[closest_centroid_index].append(point)

        if new_clusters == clusters:
            break

        clusters = new_clusters

        for i in range(num_clusters):
            if clusters[i]:
                centroid_x = sum(point[0] for point in clusters[i]) / len(clusters[i])
                centroid_y = sum(point[1] for point in clusters[i]) / len(clusters[i])
                centroids[i] = (centroid_x, centroid_y)

    return clusters

def custom_clusterization(test_sequence, num_clusters):
    n = len(test_sequence)
    dim = len(test_sequence[0])

    clusters = [[random.uniform(0, 1) for _ in range(dim)] for _ in range(num_clusters)]
    cluster_content = [[] for _ in range(num_clusters)]

    while True:
        new_cluster_content = [[] for _ in range(num_clusters)]

        for i in range(n):
            min_distance = float('inf')
            suitable_cluster = -1

            for j in range(num_clusters):
                distance = sum((test_sequence[i][q] - clusters[j][q]) ** 2 for q in range(dim)) ** 0.5

                if distance < min_distance:
                    min_distance = distance
                    suitable_cluster = j

            new_cluster_content[suitable_cluster].append(test_sequence[i])

        if new_cluster_content == cluster_content:
            break

        cluster_content = new_cluster_content

        for i in range(num_clusters):
            if cluster_content[i]:
                for q in range(dim):
                    updated_parameter = sum(point[q] for point in cluster_content[i]) / len(cluster_content[i])
                    clusters[i][q] = updated_parameter

    return cluster_content

# Згенерувати тестову послідовність
N = 1000
test_sequence = generate_test_sequence(N)

# Кількість кластерів
num_clusters = 12

# Кластеризація за алгоритмом К-середніх
k_means_clusters = k_means_clustering(test_sequence, num_clusters)
k_means_num_clusters = len(k_means_clusters)
k_means_cluster_sizes = [len(cluster) for cluster in k_means_clusters]
k_means_quality = sum(k_means_cluster_sizes) / k_means_num_clusters

# Кластеризація за вашим методом
custom_clusters = custom_clusterization(test_sequence, num_clusters)
custom_num_clusters = len(custom_clusters)
custom_cluster_sizes = [len(cluster) for cluster in custom_clusters]
custom_quality = sum(custom_cluster_sizes) / custom_num_clusters

# Побудова графіків
plt.subplot(1, 2, 1)
plt.scatter(*zip(*test_sequence), c='gray', alpha=0.5)
for i, cluster in enumerate(k_means_clusters):
    if cluster:
        x, y = zip(*cluster)
        plt.scatter(x, y, label=f'Cluster {i+1}')
plt.title('K-means Clustering')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(*zip(*test_sequence), c='gray', alpha=0.5)
for i, cluster in enumerate(custom_clusters):
    if cluster:
        x, y = zip(*cluster)
        plt.scatter(x, y, label=f'Cluster {i+1}')
plt.title('C-means Clustering')
plt.legend()

plt.tight_layout()
plt.show()

# Виведення результатів
print("K-means Clustering:")
print(f"Number of clusters: {k_means_num_clusters}")
print(f"Cluster sizes: {k_means_cluster_sizes}")
print(f"Quality: {k_means_quality}")

print("\nC-means Clustering:")
print(f"Number of clusters: {custom_num_clusters}")
print(f"Cluster sizes: {custom_cluster_sizes}")
print(f"Quality: {custom_quality}")