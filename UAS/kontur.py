import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('./bird.jpeg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray, threshold1=100, threshold2=200)

contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

img_contour = img.copy()
cv2.drawContours(img_contour, contours, -1, (0, 0, 0), thickness=2)  # warna hijau

plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.title("Deteksi Tepi ")
plt.imshow(edges, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Kontur pada Gambar Asli")
plt.imshow(cv2.cvtColor(img_contour, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.tight_layout()
plt.show()
