# K-Means Clustering for Image Segmentation
import numpy as np
import matplotlib.pyplot as plt
import cv2

class KMeansClustering:
    def __init__(self, k):  # k is the number of clusters
        self.k = k
        self.centroids = None

    @staticmethod
    def euclidean_distance(centroids, data_point):
        return np.sqrt(np.sum((centroids - data_point) ** 2, axis=1))

    def fit(self, X, max_iterations=200, tolerance=0.00005):
        # Randomly initialize centroids
        self.centroids = np.random.uniform(
            np.amin(X, axis=0), np.amax(X, axis=0), size=(self.k, X.shape[1])
        )

        for _ in range(max_iterations):
            # Assign each data point to the closest centroid
            y = np.array([np.argmin(self.euclidean_distance(self.centroids, x)) for x in X])

            # Compute new centroids
            new_centroids = np.array([
                np.mean(X[y == i], axis=0) if np.any(y == i) else self.centroids[i]
                for i in range(self.k)
            ])

            # Check for convergence
            if np.max(np.abs(self.centroids - new_centroids)) < tolerance:
                break

            self.centroids = new_centroids

        return y

# Load an image from file
image_path = r'Lab_4\Images\Sample_5.jpg'
img = cv2.imread(image_path)

if img is None:
    raise FileNotFoundError(f"Image not found at {image_path}")

# Convert BGR to RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# Convert image to float and reshape it into a 2D array
img = img.astype(np.float32) / 255.0
img_reshaped = img.reshape(-1, 3)

# Perform K-Means clustering
kmeans = KMeansClustering(k=3)
labels = kmeans.fit(img_reshaped)

# Reshape labels back to the original image shape
clustered_img = labels.reshape(img.shape[:2])

# Display the original image
plt.subplot(121)
plt.imshow(img)
plt.title("Original Image")
plt.axis("off")

# Plot the clustered image
plt.subplot(122)
plt.imshow(clustered_img, cmap='viridis')
plt.title("Clustered Image")
plt.axis("off")
plt.savefig(r'Lab_4\Task_5\Clustered_Image.png')
plt.show()
