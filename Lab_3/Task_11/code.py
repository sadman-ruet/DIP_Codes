import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

# Reading the image
loc = r'Lab_3\Images\Sample_1.jpg'
output_dir = r'Lab_3\Task_11'

f = cv2.imread(loc, cv2.IMREAD_GRAYSCALE)

# Performing FFT on the original image
F = np.fft.fft2(f)
Fshift = np.fft.fftshift(F)

# Creating a Gaussian band-reject filter
M, N = f.shape  # Shape of the image
H = np.zeros((M, N), dtype=np.float32)  # Creating a 2D array same as image size

# Band-reject filter specifications
W = 40  # Width of the band
C = 35  # Cutoff frequency

for u in range(M):
    for v in range(N):
        D = np.sqrt((u - M / 2) ** 2 + (v - N / 2) ** 2)
        if D != 0:  # Avoid division by zero
            H[u, v] = 1 - math.exp(-((D**2 - C**2) / (D * W)) ** 2)
        else:
            H[u, v] = 0  # Handle the zero division case

# Display the filter
plt.imshow(H, cmap='gray')
plt.title("Gaussian Band-Reject Filter")
plt.axis('off')
plt.show()

# Applying the filter in the frequency domain
Gshift = Fshift * H
G = np.fft.ifftshift(Gshift)
g = np.abs(np.fft.ifft2(G))

# Normalize pixel values to the range [0, 255]
g_normalized = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX)

# Convert to uint8 before saving
g_uint8 = np.uint8(g_normalized)

# Saving the filtered image
output_path = os.path.join(output_dir, 'Band_Reject_Gaussian_Filtered.png')
cv2.imwrite(output_path, g_uint8)

print(f"Filtered image saved at {output_path}")