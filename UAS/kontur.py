import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('./bird.jpeg', cv2.IMREAD_GRAYSCALE)
edges = cv2.Canny(img, 100, 200)

def find_contours_manual(binary_img):
    visited = np.zeros_like(binary_img, dtype=bool)
    h, w = binary_img.shape
    contours = []

    # Arah gerakan 8-tetangga
    directions = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),          (0, 1),
                  (1, -1),  (1, 0), (1, 1)]

    def trace_contour(y, x):
        contour = [(y, x)]
        visited[y, x] = True
        while True:
            found = False
            for dy, dx in directions:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    if binary_img[ny, nx] == 255 and not visited[ny, nx]:
                        contour.append((ny, nx))
                        visited[ny, nx] = True
                        y, x = ny, nx
                        found = True
                        break
            if not found:
                break
        return contour

    # Loop setiap piksel
    for i in range(h):
        for j in range(w):
            if binary_img[i, j] == 255 and not visited[i, j]:
                c = trace_contour(i, j)
                if len(c) > 10:  # abaikan kontur kecil/noise
                    contours.append(c)
    return contours

contours = find_contours_manual(edges)

output_img = np.ones((*edges.shape, 3), dtype=np.uint8) * 255  # putih

for contour in contours:
    for y, x in contour:
        output_img[y, x] = [0, 0, 0]  # hitam

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Edge Deteksi (Canny)")
plt.imshow(edges, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Kontur ")
plt.imshow(output_img)
plt.axis('off')

plt.tight_layout()
plt.show()
