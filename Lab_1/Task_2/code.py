import numpy as np
import cv2
import matplotlib.pyplot as plt

# Input image path
image_path = 'Lab_1/Images/Sample_1.png'
save_path = 'Lab_1/Task_2/output.png'

# Read the input image
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f"File '{image_path}' could not be read. Check if the path is correct.")

# Convert from BGR to RGB so we can plot using matplotlib
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Disable x & y axis
plt.axis('off')

# Show the original image
plt.imshow(img)
plt.show()

# Get the image shape
rows, cols, dim = img.shape

# Transformation matrix for translation
M = np.float32([[1, 0, 50],
                [0, 1, 50],
                [0, 0, 1]])

# Apply a perspective transformation to the image
translated_img = cv2.warpPerspective(img, M, (cols, rows))

# Save the translated image
cv2.imwrite(save_path, cv2.cvtColor(translated_img, cv2.COLOR_RGB2BGR))

# Disable x & y axis
plt.axis('off')

# Show the resulting image
plt.imshow(translated_img)
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
