import cv2
import numpy as np
import matplotlib.pyplot as plt

# Input image path
image_path = 'Lab_2\Images\Sample_2.jpg'
save_path_filtered = 'Lab_2/Task_3/Laplacian_Filtered_Sample_1.png'

# Read the input image in grayscale
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(f"File '{image_path}' could not be read. Check if the path is correct.")

# Define the Laplacian kernel
laplacian_kernel = np.array([
    [1, 1, 1],
    [1, -8, 1],
    [1, 1, 1]
], dtype=np.float32)

# Apply the Laplacian filter using OpenCV
laplacian_filtered = cv2.filter2D(img, ddepth=-1, kernel=laplacian_kernel)

# Sharpen the image by adding the Laplacian response
c = -1  # Control the amount of sharpening
sharpened_img = cv2.addWeighted(img, 1, laplacian_filtered, c, 0)

# Clip pixel values to the valid range
sharpened_img = np.clip(sharpened_img, 0, 255).astype(np.uint8)

# Save the filtered image
cv2.imwrite(save_path_filtered, sharpened_img)

# Display the original and filtered images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(sharpened_img, cmap='gray')
plt.title('Laplacian Sharpened Image')
plt.axis('off')

plt.show()