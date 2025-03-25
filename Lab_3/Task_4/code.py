import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

# Read the input image
loc = r'Lab_3\Images\Sample_1.jpg'
output_dir = r'Lab_3\Task_4'
f = cv2.imread(loc, cv2.IMREAD_GRAYSCALE)
if f is None:
    raise FileNotFoundError(f"Image could not be read. Check if the path is correct.")

# Performing FFT on the original image
F = np.fft.fft2(f)
Fshift = np.fft.fftshift(F)

# Create a high-pass filter
rows, cols = f.shape
center_row, center_col = rows // 2, cols // 2  # Center of the image
radius = 50  # Radius for the high-pass filter

# Creating a mask with a circular region set to 0 (low frequencies removed) and the rest set to 1
H = np.ones((rows, cols), np.uint8)
cv2.circle(H, (center_col, center_row), radius, (0, 0), -1)  # Set the center region to 0

# Apply the mask to the shifted DFT
Gshift = Fshift * H

# Perform inverse FFT to get the filtered image
G = np.fft.ifftshift(Gshift)
g = np.abs(np.fft.ifft2(G))

# Convert the result to uint8 (clip values between 0 and 255)
g = np.uint8(np.clip(g, 0, 255))

# Saving the filtered image
output_path = os.path.join(output_dir, 'High_Pass_Ideal_filter.png')
cv2.imwrite(output_path, g)

print(f"Filtered image saved at {output_path}")

# Optionally, display the original and filtered images for visualization
plt.figure(figsize=(12, 6))

plt.subplot(121)
plt.imshow(f, cmap='gray')
plt.title("Original Image")
plt.axis('off')

plt.subplot(122)
plt.imshow(g, cmap='gray')
plt.title("High Pass Filtered Image")
plt.axis('off')

plt.savefig(output_path)
# Show the figure
plt.show()
