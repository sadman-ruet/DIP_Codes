import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Open the image (convert to grayscale if it's in color)
f = cv2.imread(r'Lab_3\Images\Sample_1.jpg', 0)
output_dir = r'Lab_3/Task_3'

if f is None:
    raise FileNotFoundError(f"Image could not be read. Check if the path is correct.")

# Transform image into frequency domain and shift zero-frequency component to the center
F = np.fft.fft2(f)
Fshift = np.fft.fftshift(F)

# Create Gaussian Low Pass Filter
M, N = f.shape
H = np.zeros((M, N), dtype=np.float32)
D0 = 20  # Cutoff frequency
n = 5     # Order of the filter

# Apply Gaussian Low Pass filter formula
for u in range(M):
    for v in range(N):
        D = np.sqrt((u - M / 2) ** 2 + (v - N / 2) ** 2)
        H[u, v] = np.exp(-D ** 2 / (2 * D0 * D0))

# Apply the filter in frequency domain
Gshift = Fshift * H
G = np.fft.ifftshift(Gshift)

# Perform inverse FFT to get the filtered image in spatial domain
g = np.abs(np.fft.ifft2(G))

# Convert the result to uint8 (clip values between 0 and 255)
g = np.uint8(np.clip(g, 0, 255))

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Optionally, display the original and filtered images for visualization
plt.figure(figsize=(12, 6))

plt.subplot(121)
plt.imshow(f, cmap='gray')
plt.title("Original Image")
plt.axis('off')

plt.subplot(122)
plt.imshow(g, cmap='gray')
plt.title("Gaussian Filtered Image")
plt.axis('off')

# Saving the filtered image
output_path = os.path.join(output_dir, 'Low_Pass_Gaussian_Filtered.png')
plt.savefig(output_path)
# Show the figure
plt.show()
