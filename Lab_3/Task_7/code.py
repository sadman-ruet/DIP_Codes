import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

# Read the input image
loc = r'Lab_3\Images\Sample_1.jpg'
output_dir = r'Lab_3\Task_7'
f = cv2.imread(loc, cv2.IMREAD_GRAYSCALE)

# Performing FFT on the original image
F = np.fft.fft2(f)
Fshift = np.fft.fftshift(F)

# Create an ideal band-pass filter
rows, cols = f.shape  # Use f.shape instead of image.shape
center_row, center_col = rows // 2, cols // 2

# Specify the radius of the ideal band-pass filter
radius1 = 30  # Inner radius (low frequency cut-off)
radius2 = 50  # Outer radius (high frequency cut-off)

# Create a mask with a circular region set to 1 and the rest set to 0
H1 = np.ones((rows, cols), np.uint8)
cv2.circle(H1, (center_col, center_row), radius1, (0, 0), -1)  # Set the inner region to 0

H2 = np.ones((rows, cols), np.uint8)
cv2.circle(H2, (center_col, center_row), radius2, (0, 0), -1)  # Set the outer region to 0

# Band-pass filter is the difference between H2 and H1
H = H2 - H1

# Ensure the filter values are between 0 and 1
H = np.clip(H, 0, 1)

# Apply the mask to the shifted DFT
Gshift = Fshift * H
G = np.fft.ifftshift(Gshift)

# Perform the inverse FFT to get the filtered image
g = np.abs(np.fft.ifft2(G))

# Convert to uint8 (clip values between 0 and 255)
g = np.uint8(np.clip(g, 0, 255))

# Saving the filtered image
output_path = os.path.join(output_dir, 'Band_Pass_Ideal_filter.png')
cv2.imwrite(output_path, g)

print(f"Filtered image saved at {output_path}")
