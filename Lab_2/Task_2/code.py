import cv2
import numpy as np
import matplotlib.pyplot as plt

# Input image path
image_path = 'Lab_2/Images/Sample_1.png'
save_path_filtered = 'Lab_2/Task_2/output.png'

# Read the input image
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f"File '{image_path}' could not be read. Check if the path is correct.")

M, N, _ = img.shape  # M = image height, N = image width
kernel_size = 50  # You can change this value as needed
sigma = 3.5  # Standard deviation for Gaussian kernel

# Function to generate Gaussian kernel
def gaussian_kernel(size, sigma):
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / np.sum(kernel)

# Generate Gaussian kernel
gauss_kernel = gaussian_kernel(kernel_size, sigma)
print('Gaussian kernel size:', gauss_kernel.shape)

# Apply Gaussian filter using OpenCV
filtered_image = cv2.filter2D(src=img, ddepth=-1, kernel=gauss_kernel)

# Save the filtered image
cv2.imwrite(save_path_filtered, filtered_image)

# Display the filtered image
plt.imshow(cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB))
plt.title('Gaussian Filtered Image')
plt.axis('off')
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
