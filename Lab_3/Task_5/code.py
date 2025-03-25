import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Reading the image
loc = r'Lab_3\Images\Sample_1.jpg'
output_dir = r'Lab_3\Task_5'
f = cv2.imread(loc, cv2.IMREAD_GRAYSCALE)
if f is None:
    raise FileNotFoundError(f"Image could not be read. Check if the path is correct.")

# Performing FFT on the original image
F = np.fft.fft2(f)
Fshift = np.fft.fftshift(F)

M, N = f.shape
HPF = np.zeros((M, N), dtype=np.float32)  # High-pass filter initialization

D0 = 50  # Cutoff frequency
n = 2     # Filter order

# Applying the High Pass Butterworth filter in the frequency domain
for u in range(M):
    for v in range(N):
        D = np.sqrt((u - M / 2) ** 2 + (v - N / 2) ** 2)
        # Avoid zero division
        if D != 0:
            HPF[u, v] = 1 / (1 + (D0 / D) ** (2 * n))
        else:
            HPF[u, v] = 0

# Apply the filter in the frequency domain
Gshift = Fshift * HPF

# Perform inverse FFT to get the filtered image in spatial domain
G = np.fft.ifftshift(Gshift)
g = np.abs(np.fft.ifft2(G))

# Convert the result to uint8 (clip values between 0 and 255)
g = np.uint8(np.clip(g, 0, 255))

# Saving the filtered image
output_path = os.path.join(output_dir, 'High_Pass_Butterworth_Filtered.png')
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
plt.title("High Pass Butterworth Filtered Image")
plt.axis('off')
plt.savefig(output_path)
# Show the figure
plt.show()
