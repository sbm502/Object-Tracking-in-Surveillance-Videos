import cv2
import matplotlib.pyplot as plt

# Load image
img = cv2.imread('data/messi5.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Harris Corner Detection
corners = cv2.cornerHarris(gray, 2, 3, 0.04)
img[corners > 0.01 * corners.max()] = [0, 0, 255]

# Show Image
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Detected Corners")
plt.show()
