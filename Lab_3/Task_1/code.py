import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

# Read the input image
loc = r'Lab_3\Images\Sample_1.jpg'  # Change the image name accordingly
output_dir = r'Lab_3/Task_1'

# Check if the directory exists, create if it doesn't
os.makedirs(output_dir, exist_ok=True)

# Read the image in grayscale
f = cv2.imread(loc, cv2.IMREAD_GRAYSCALE)
if f is None:
    raise FileNotFoundError(f"Image '{loc}' could not be read. Check if the path is correct.")

# Performing FFT on the original image
F = np.fft.fft2(f)
Fshift = np.fft.fftshift(F)

# Create an ideal low-pass filter
rows, cols = f.shape
center_row, center_col = rows // 2, cols // 2

# Radius of the ideal low-pass filter
radius = 30

# Creating a mask with a circular region set to 1 and the rest set to 0
H = np.zeros((rows, cols), np.uint8)
cv2.circle(H, (center_col, center_row), radius, (1, 1), -1)

# Apply the mask to the shifted DFT
Gshift = Fshift * H

# Inverse FFT to get the filtered image
G = np.fft.ifftshift(Gshift)
g = np.abs(np.fft.ifft2(G))

# Convert the result to uint8 for saving
g = np.uint8(np.clip(g, 0, 255))



# Display the original and filtered images
plt.figure(figsize=(10, 5))

plt.subplot(121)
plt.imshow(f, cmap='gray')
plt.title("Original Image")
plt.axis('off')

plt.subplot(122)
plt.imshow(g, cmap='gray')
plt.title("Filtered Image with Ideal Low Pass Filter")
plt.axis('off')
# Saving the filtered image
output_path = os.path.join(output_dir, 'Low_Pass_Ideal_Filtered_Image.png')
plt.savefig(output_path)

# Show the figure
plt.show()

print(f"Filtered image saved at {output_path}")
