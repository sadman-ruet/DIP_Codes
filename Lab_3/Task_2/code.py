import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Reading the image
loc = r'Lab_3\Images\Sample_1.jpg'  # Path to the image
output_dir = r'Lab_3/Task_2'  # Path to save the output image

f = cv2.imread(loc, cv2.IMREAD_GRAYSCALE)
if f is None:
    raise FileNotFoundError(f"Image '{loc}' could not be read. Check if the path is correct.")

# Performing FFT on the original image
F = np.fft.fft2(f)
Fshift = np.fft.fftshift(F)

# Creating the Butterworth Low Pass Filter
M, N = f.shape  # Shape of the image
H = np.zeros((M, N), dtype=np.float32)  # Create a 2D array with the same size as the image

D0 = 10  # Cutoff frequency
n = 3     # Order of the filter

# Creating the Butterworth filter
for u in range(M):
    for v in range(N):
        D = np.sqrt((u - M / 2)**2 + (v - N / 2)**2)
        H[u, v] = 1 / (1 + (D / D0)**(2 * n))  # Butterworth filter formula

# Apply the filter in the frequency domain
Gshift = Fshift * H
G = np.fft.ifftshift(Gshift)

# Perform the inverse FFT to get the filtered image in the spatial domain
g = np.abs(np.fft.ifft2(G))  # Get the magnitude of the inverse FFT result

# Convert the filtered image to uint8 (clip values between 0 and 255)
g = np.uint8(np.clip(g, 0, 255))

# Saving the filtered image
output_path = os.path.join(output_dir, 'Low_Pass_Butterworth_Filtered.png')

# Optionally, display the original and filtered images for visualization
plt.figure(figsize=(12, 6))

plt.subplot(121)
plt.imshow(f, cmap='gray')
plt.title("Original Image")
plt.axis('off')


plt.subplot(122)
plt.imshow(g, cmap='gray')
plt.title("Butterworth Filtered Image")
plt.axis('off')
plt.savefig(output_path)
plt.show()
