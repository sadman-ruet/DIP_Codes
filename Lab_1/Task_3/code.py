import numpy as np
import os
import cv2 as cv

# Input image path
image_path = 'Lab_1/Images/Sample_1.png'
save_path_scaled = 'Lab_1/Task_3/Scaled_Sample_1.png'
save_path_shrunk = 'Lab_1/Task_3/Shrunk_Sample_1.png'

# Read the input image
imag = cv.imread(image_path)
if imag is None:
    raise FileNotFoundError(f"File '{image_path}' could not be read. Check if the path is correct.")

rows, cols, _ = imag.shape

# Scaling (Enlarging)
M_scale = np.float32([[2, 0, 0], [0, 2, 0], [0, 0, 1]])
dst_scaled = cv.warpPerspective(imag, M_scale, (cols * 2, rows * 2))
cv.imwrite(save_path_scaled, dst_scaled)
cv.imshow('Scaled Image', dst_scaled)
cv.waitKey(0)
cv.destroyAllWindows()

# Shrinking
M_shrink = np.float32([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 1]])
dst_shrunk = cv.warpPerspective(imag, M_shrink, (cols // 2, rows // 2))
cv.imwrite(save_path_shrunk, dst_shrunk)
cv.imshow('Shrunk Image', dst_shrunk)
cv.waitKey(0)
cv.destroyAllWindows()
