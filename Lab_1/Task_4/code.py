import numpy as np
import cv2
import matplotlib.pyplot as plt

# Input image path
image_path = 'Lab_1/Images/Sample_1.png'
save_path_sheared_x = 'Lab_1/Task_4/Sheared_X_Sample_1.png'
save_path_sheared_y = 'Lab_1/Task_4/Sheared_Y_Sample_1.png'

# Read the input image
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f"File '{image_path}' could not be read. Check if the path is correct.")

# Convert from BGR to RGB so we can plot using matplotlib
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()

# Get the image shape
rows, cols, dim = img.shape

# Shearing applied to x-axis
M_x = np.float32([[1, 0.5, 0],
                  [0, 1, 0],
                  [0, 0, 1]])
sheared_x_img = cv2.warpPerspective(img, M_x, (int(cols * 1.5), int(rows * 1.5)))
cv2.imwrite(save_path_sheared_x, cv2.cvtColor(sheared_x_img, cv2.COLOR_RGB2BGR))

plt.axis('off')
plt.imshow(sheared_x_img)
plt.show()

# Shearing applied to y-axis
M_y = np.float32([[1, 0, 0],
                  [0.5, 1, 0],
                  [0, 0, 1]])
sheared_y_img = cv2.warpPerspective(img, M_y, (int(cols * 1.5), int(rows * 1.5)))
cv2.imwrite(save_path_sheared_y, cv2.cvtColor(sheared_y_img, cv2.COLOR_RGB2BGR))

plt.axis('off')
plt.imshow(sheared_y_img)
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
