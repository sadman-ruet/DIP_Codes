import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Open the image
f = cv2.imread(r'Lab_3\Images\Sample_1.jpg', 0)
output_dir = r'Lab_3\Task_6'

# Transform image into frequency domain and shift
F = np.fft.fft2(f)
Fshift = np.fft.fftshift(F)

# Gaussian High Pass Filter
M, N = f.shape
H = np.zeros((M, N), dtype=np.float32)
D0 = 50  # Cutoff frequency
n = 2     # Filter order

# Create the Gaussian Low Pass Filter
for u in range(M):
    for v in range(N):
        D = np.sqrt((u - M / 2) ** 2 + (v - N / 2) ** 2)
        H[u, v] = np.exp(-D ** 2 / (2 * D0 ** 2))

# High Pass Filter is 1 - Low Pass Filter
HPF = 1 - H

# Apply the filter in the frequency domain
Gshift = Fshift * HPF

# Perform inverse FFT to get the filtered image in spatial domain
G = np.fft.ifftshift(Gshift)
g = np.abs(np.fft.ifft2(G))

# Convert the result to uint8 (clip values between 0 and 255)
g = np.uint8(np.clip(g, 0, 255))

# Saving the filtered image
output_path = os.path.join(output_dir, 'High_Pass_Gaussian_Filtered.png')
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
plt.title("High Pass Gaussian Filtered Image")
plt.axis('off')
plt.savefig(output_path)

# Show the figure
plt.show()