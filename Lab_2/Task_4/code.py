import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define the image and save paths
image_path = 'Lab_2/Images/Sample_2.jpg'
save_path_filtered = 'Lab_2/Task_4/Filtered_Sample_2.jpg'

# Read the image
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f"File '{image_path}' could not be read. Check if the path is correct.")

M, N, _ = img.shape  # M = image height, N = image width

# Kernel size for blurring the image
kernel_size = 5  # You can adjust this value
kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)

# Blurring the image using filter2D
blurred_img = cv2.filter2D(img, -1, kernel)

# Unsharp Masking: mask = original image - blurred image
mask = cv2.subtract(img, blurred_img)

# Apply unsharp masking (using k = 1)
k = 1
unsharp_img = cv2.add(img, k * mask)

# High Boost Filtering (using k = 3)
k_high_boost = 3
high_boost_img = cv2.add(img, k_high_boost * mask)

# Save the images
cv2.imwrite(save_path_filtered, high_boost_img)

# Display results
plt.subplot(121)
plt.imshow(cv2.cvtColor(unsharp_img, cv2.COLOR_BGR2RGB))
plt.title('Unsharp Masking')
plt.axis('off')

plt.subplot(122)
plt.imshow(cv2.cvtColor(high_boost_img, cv2.COLOR_BGR2RGB))
plt.title('High Boost Filtering')
plt.axis('off')
plt.savefig('Lab_2/Task_4/output.png')
plt.show()
