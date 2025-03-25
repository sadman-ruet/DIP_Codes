# Edge Detection using Canny Algorithm (Manual Implementation)
import cv2
import os
import numpy as np

def canny_edge_detector(image_path, low_threshold, high_threshold):
    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Apply Gaussian blur to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    # Compute gradients using Sobel operators
    gradient_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)

    # Compute gradient magnitude and direction
    magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    direction = np.arctan2(gradient_y, gradient_x) * (180 / np.pi)

    # Normalize direction values to [0, 180]
    direction = (direction + 180) % 180

    # Non-Maximum Suppression
    suppressed = np.zeros_like(magnitude)

    for i in range(1, magnitude.shape[0] - 1):
        for j in range(1, magnitude.shape[1] - 1):
            angle = direction[i, j]

            if (0 <= angle < 22.5) or (157.5 <= angle <= 180):  # Horizontal edges
                neighbors = [magnitude[i, j - 1], magnitude[i, j + 1]]
            elif (22.5 <= angle < 67.5):  # +45 degree diagonal
                neighbors = [magnitude[i - 1, j + 1], magnitude[i + 1, j - 1]]
            elif (67.5 <= angle < 112.5):  # Vertical edges
                neighbors = [magnitude[i - 1, j], magnitude[i + 1, j]]
            elif (112.5 <= angle < 157.5):  # -45 degree diagonal
                neighbors = [magnitude[i - 1, j - 1], magnitude[i + 1, j + 1]]
            else:
                neighbors = []

            if magnitude[i, j] >= max(neighbors, default=0):  # Keep local maxima
                suppressed[i, j] = magnitude[i, j]

    # Double thresholding and edge tracking by hysteresis
    edges = np.zeros_like(suppressed, dtype=np.uint8)
    strong_edge = 255
    weak_edge = 75

    strong_pixels = suppressed >= high_threshold
    weak_pixels = (suppressed >= low_threshold) & (suppressed < high_threshold)

    edges[strong_pixels] = strong_edge
    edges[weak_pixels] = weak_edge

    # Edge tracking by hysteresis
    for i in range(1, edges.shape[0] - 1):
        for j in range(1, edges.shape[1] - 1):
            if edges[i, j] == weak_edge:
                if (
                    (edges[i + 1, j] == strong_edge) or (edges[i - 1, j] == strong_edge) or
                    (edges[i, j + 1] == strong_edge) or (edges[i, j - 1] == strong_edge) or
                    (edges[i - 1, j - 1] == strong_edge) or (edges[i - 1, j + 1] == strong_edge) or
                    (edges[i + 1, j - 1] == strong_edge) or (edges[i + 1, j + 1] == strong_edge)
                ):
                    edges[i, j] = strong_edge
                else:
                    edges[i, j] = 0  # Suppress weak edges not connected to strong ones

    return edges

# Set image path and thresholds
image_path = r'Lab_4\Images\Sample_3.jpg'
low_threshold = 35
high_threshold = 150

# Run Canny Edge Detector
edges = canny_edge_detector(image_path, low_threshold, high_threshold)

# Output directory
output_dir = r"Lab_4\Task_3"
os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists

# Save the result
output_path = os.path.join(output_dir, 'edge_detection_canny.png')
cv2.imwrite(output_path, edges)

print(f"Saved: {output_path}")
