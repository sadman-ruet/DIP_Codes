import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

# Creating Gaussian function
def gaussian_kernel(size, sigma=1):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]  # Creates a 2D grid of coordinates
    normal = 1 / (2.0 * np.pi * sigma**2)
    g = np.exp(-((x**2 + y**2) / (2.0 * sigma**2))) * normal
    return g

gauss_kernel = gaussian_kernel(3, sigma=1)

# Harris corner response function
def corner_response(image, k, G):
    # Compute first derivatives
    dx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    dy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)

    # Gaussian filtering
    A = cv2.filter2D(dx * dx, ddepth=-1, kernel=G)
    B = cv2.filter2D(dy * dy, ddepth=-1, kernel=G)
    C = cv2.filter2D(dx * dy, ddepth=-1, kernel=G)

    # Compute corner response at all pixels
    return (A * B - (C * C)) - k * (A + B) * (A + B)

# Harris corner detection function
def detect_corners(image, k, G, thresh):
    # Convert RGB image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute Harris response
    harris_response = corner_response(gray, k, G)

    # Dilate the response to find local maxima
    harris_response_dilated = cv2.dilate(harris_response, None)

    # Mark corners on the original image
    corner_threshold = thresh * harris_response_dilated.max()
    image[harris_response_dilated > corner_threshold] = [0, 255, 255]  

    return image

# Hyperparameters
k = 0.04
thresh = 0.005

# Load an image
image = cv2.imread(r'Lab_5\Images\Sample_1.jpg')

# Ensure the image is loaded properly
if image is None:
    raise ValueError("Error: Image not found. Check the file path!")

# Detect corners manually
result_image = detect_corners(image, k, gauss_kernel, thresh)

# Output Directory
output_dir=r"Lab_5/Task_1"
# Display the result
plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title("Harris Corner Detection")
plt.savefig(output_dir + "/Harris_Corner_Detection.png")
plt.show()
