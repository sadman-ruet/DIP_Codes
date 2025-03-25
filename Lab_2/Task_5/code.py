import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define the image and save paths
image_path = 'Lab_2/Images/Sample_2.jpg'
save_path_filtered = 'Lab_2/Task_5/Negative_Sample_2_Comparison.jpg'

# Read the image in grayscale
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(f"File '{image_path}' could not be read. Check if the path is correct.")

M, N = img.shape  # M = image height, N = image width
L = 255  # Maximum intensity value for 8-bit images

# Initialize the output image for the negative transformation
g = np.zeros((M, N), dtype=np.float32)

# Negative transformation
for i in range(M):
    for j in range(N):
        g[i, j] = L - 1 - img[i, j]

# Create a single figure with subplots for images and histograms
plt.figure(figsize=(12, 8))

# Original Image and Histogram
plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray')
plt.title("Original Image")
plt.axis('off')

plt.subplot(2, 2, 2)
plt.hist(img.ravel(), bins=256, range=[0, 256], color='blue')
plt.title('Histogram of Original Image')
plt.xlabel('Intensity Labels')
plt.ylabel('Frequency')

# Negative Image and Histogram
plt.subplot(2, 2, 3)
plt.imshow(g, cmap='gray')
plt.title('Negative Image')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.hist(g.ravel(), bins=256, range=[0, 256], color='blue')
plt.title('Histogram of Negative Image')
plt.xlabel('Intensity Labels')
plt.ylabel('Frequency')

# Save the figure
plt.tight_layout()
plt.savefig(save_path_filtered)

# Show the figure
plt.show()
