import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Reading the image
loc = r'C:\Users\User\Documents\WorksSpace\Lab Report Code\Lab_2\Images\Sample_1.png'
output_dir = r'Lab_3\Task_8'
f = cv2.imread(loc, cv2.IMREAD_GRAYSCALE)

# Performing FFT on the original image
F = np.fft.fft2(f)
Fshift = np.fft.fftshift(F)

# Creating a Butterworth bandpass filter
M, N = f.shape  # Shape of the image
H = np.zeros((M, N), dtype=np.float32)  # Create a 2D array same as image size

# Bandpass filter specifications
n = 3  # Order of the filter
W = 30  # Width of the circle
C = 40  # Cutoff value

# Butterworth Bandpass filter formula
for u in range(M):
    for v in range(N):
        D = np.sqrt((u - M / 2) ** 2 + (v - N / 2) ** 2)  # Distance from the center
        # To prevent division by zero at the center, set D != 0
        if D != 0:
            H[u, v] = 1 - (1 / (1 + ((D * W) / ((D**2) - (C**2))))**(2 * n))
        else:
            H[u, v] = 0

# Apply the filter in the frequency domain
Gshift = Fshift * H
G = np.fft.ifftshift(Gshift)
g = np.abs(np.fft.ifft2(G))

# Normalize pixel values to the range [0, 255]
g_normalized = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX)

# Saving filtered image
output_path = os.path.join(output_dir, 'Band_Pass_Butterworth_Filtered.png')
cv2.imwrite(output_path, g_normalized)

print(f"Filtered image saved at {output_path}")
