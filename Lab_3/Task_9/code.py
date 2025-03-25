import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Read the input image
loc = r'Lab_3\Images\Sample_1.jpg'
output_dir = r'Lab_3\Task_9'
f = cv2.imread(loc, cv2.IMREAD_GRAYSCALE)

# Perform FFT on the original image
F = np.fft.fft2(f)
Fshift = np.fft.fftshift(F)

# Create an ideal band-reject filter
rows, cols = f.shape  # Use f.shape instead of image.shape
center_row, center_col = rows // 2, cols // 2

# Specify the radius of the ideal band-reject filter
radius1 = 30  # Inner radius
radius2 = 50  # Outer radius

# Create a mask with a circular region set to 1 (to reject), and the rest set to 0
H1 = np.ones((rows, cols), np.uint8)   # Full mask initialized to 1
cv2.circle(H1, (center_col, center_row), radius1, (0, 0), -1)  # Inner circle set to 0

H2 = np.ones((rows, cols), np.uint8)   # Outer mask initialized to 1
cv2.circle(H2, (center_col, center_row), radius2, (1, 1), -1)  # Outer circle set to 1

# Create the band-reject filter by subtracting H1 from H2
H = H2 - H1

# Ensure the filter values are between 0 and 1 (logical masking)
H = np.clip(H, 0, 1)

# Visualizing the Ideal Band Reject Filter
plt.imshow(H, cmap='gray')
plt.title("Ideal Band Reject Filter")
plt.axis('off')
plt.show()

# Apply the mask to the shifted DFT
Gshift = Fshift * H
G = np.fft.ifftshift(Gshift)

# Inverse FFT to get the filtered image
g = np.abs(np.fft.ifft2(G))

# Normalize the image to the range [0, 255] for visualization and saving
g_normalized = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX)

# Saving the filtered image
output_path = os.path.join(output_dir, 'Band_Reject_Ideal_Filtered.png')
cv2.imwrite(output_path, g_normalized)

print(f"Filtered image saved at {output_path}")
