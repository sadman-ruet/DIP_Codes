import cv2
import numpy as np
import matplotlib.pyplot as plt

# Input image path
image_path = 'Lab_2/Images/Sample_1.png'
save_path_filtered = 'Lab_2/Task_1/Filtered_Sample_1.png'

# Read the input image
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f"File '{image_path}' could not be read. Check if the path is correct.")

M, N, _ = img.shape  # M = image height, N = image width
kernel_size = 50  # You can change this value as needed

# Create the box filter kernel
kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)

# Apply the box filter using OpenCV
filtered_image = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)

# Save the filtered image
cv2.imwrite(save_path_filtered, filtered_image)

# Display the original image
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')
plt.show()

# Display the filtered image
plt.imshow(cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB))
plt.title('Filtered Image')
plt.axis('off')
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
