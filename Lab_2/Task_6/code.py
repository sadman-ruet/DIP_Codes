import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import os

# Create the directory for saving images if it doesn't exist
save_dir = 'Lab_2/Task_6'
os.makedirs(save_dir, exist_ok=True)

# Read the image in grayscale
img = cv2.imread(r'Lab_2\Images\Sample_2.jpg', cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("Image could not be read. Check the path.")

M, N = img.shape  # M = image height, N = image width

# Create empty arrays for the transformations
g = np.zeros((M, N), dtype=np.float32)  # Log transformed image
inverse_log = np.zeros((M, N), dtype=np.float32)  # Inverse log transformed image

# Calculate constant for logarithmic transformation
c = 255 / (math.log10(1 + np.max(img)))

# Apply log transformation
for i in range(M):
    for j in range(N):
        g[i, j] = c * math.log10(1 + img[i, j])

# Apply inverse log transformation
for i in range(M):
    for j in range(N):
        inverse_log[i, j] = 10**(img[i, j] / c) - 1

# Create a figure for all plots
plt.figure(figsize=(15, 10))

# Plot original image and histogram
plt.subplot(3, 2, 1)
plt.imshow(img, cmap='gray')
plt.title("Original Image")
plt.axis('off')

plt.subplot(3, 2, 2)
plt.hist(img.ravel(), bins=256, range=[0, 256], color='blue')
plt.title('Histogram of Original Image')
plt.xlabel('Intensity Labels')
plt.ylabel('Frequency')

# Plot log transformed image and histogram
plt.subplot(3, 2, 3)
plt.imshow(g, cmap='gray')
plt.title('Log Transformed Image')
plt.axis('off')

plt.subplot(3, 2, 4)
plt.hist(g.ravel(), bins=256, range=[0, 256], color='blue')
plt.title('Histogram of Log Transformed Image')
plt.xlabel('Intensity Labels')
plt.ylabel('Frequency')

# Plot inverse log transformed image and histogram
plt.subplot(3, 2, 5)
plt.imshow(inverse_log, cmap='gray')
plt.title('Inverse Log Transformed Image')
plt.axis('off')

plt.subplot(3, 2, 6)
plt.hist(inverse_log.ravel(), bins=256, range=[0, 256], color='blue')
plt.title('Histogram of Inverse Log Transformed Image')
plt.xlabel('Intensity Labels')
plt.ylabel('Frequency')

# Save the figure
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'Log_and_Inverse_Log_Transformation_Results.png'))

# Show the figure
plt.show()

print(f"Figure saved in {save_dir}")
