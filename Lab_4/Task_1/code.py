# Line Detection
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

# Making filter function
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
image_path = r'Lab_4\Images\Sample_1.png'
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Ensure image is loaded
if img is None:
    raise FileNotFoundError(f"Image not found at {image_path}")

# Output directory
output_dir = r'Lab_4\Task_1'
os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists

# Define kernels
kernels = {
    "horizontal": np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]], dtype=np.float32),
    "vertical": np.array([[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]], dtype=np.float32),
    "+45": np.array([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]], dtype=np.float32),
    "-45": np.array([[-1, -1, 2], [-1, 2, -1], [2, -1, -1]], dtype=np.float32),
}

# Apply filters and save results
for direction, kernel in kernels.items():
    output_image = filter(img, kernel)
    output_path = os.path.join(output_dir, f'Line_detection_{direction}.png')
    cv2.imwrite(output_path, output_image)
    print(f"Saved: {output_path}")
