# Edge Detection
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

# Filter function
def filter(image, kernel):
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    result = np.zeros_like(image, dtype=np.float32)

    for i in range(image_height - kernel_height + 1):
        for j in range(image_width - kernel_width + 1):
            area = image[i:i + kernel_height, j:j + kernel_width]
            conv = np.sum(area * kernel)  # Using element-wise multiplication
            result[i + kernel_height // 2, j + kernel_width // 2] = conv  # Correct indexing

    return np.clip(result, 0, 255).astype(np.uint8)  # Ensure valid pixel values

# Set image path
image_path = r'Lab_4\Images\Sample_2.jpg'
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Ensure image is loaded
if img is None:
    raise FileNotFoundError(f"Image not found at {image_path}")

# Output directory
output_dir = r'Lab_4\Task_2'
os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists

# Edge Detection using Laplacian Kernel
laplacian_kernel = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
], dtype=np.float32)

edge_detection_laplacian = filter(img, laplacian_kernel)

# Save Laplacian result
output_path = os.path.join(output_dir, 'edge_detection_laplacian.png')
cv2.imwrite(output_path, edge_detection_laplacian)
print(f"Saved: {output_path}")

# Edge Detection using Sobel Kernels
sobel_horizontal = np.array([
    [1,  2,  1],
    [0,  0,  0],
    [-1, -2, -1]
], dtype=np.float32)

sobel_vertical = np.array([
    [ 1,  0, -1],
    [ 2,  0, -2],
    [ 1,  0, -1]
], dtype=np.float32)

# Apply Sobel filters
edge_horizontal = filter(img, sobel_horizontal)
edge_vertical = filter(img, sobel_vertical)

# Combine Sobel edges (gradient magnitude)
edge_detection_sobel = np.sqrt(edge_horizontal.astype(np.float32) ** 2 + edge_vertical.astype(np.float32) ** 2)
edge_detection_sobel = np.clip(edge_detection_sobel, 0, 255).astype(np.uint8)  # Convert back to valid range

# Save Sobel result
output_path = os.path.join(output_dir, 'edge_detection_sobel.png')
cv2.imwrite(output_path, edge_detection_sobel)
print(f"Saved: {output_path}")