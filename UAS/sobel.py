import cv2
import numpy as np
import matplotlib.pyplot as plt

def edgeDetectionSobel(img, threshold=100):
    sX = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float32)

    sY = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ], dtype=np.float32)

    h, w = img.shape
    edgeImg = np.zeros((h-2, w-2), dtype=np.float32)
    Gx_img = np.zeros((h-2, w-2), dtype=np.float32)
    Gy_img = np.zeros((h-2, w-2), dtype=np.float32)

    for i in range(1, h-1):
        for j in range(1, w-1):
            matrx =img[i-1:i+2, j-1:j+2]
            gX = np.sum(matrx*sX)
            gY = np.sum(matrx*sY)
            g = np.sqrt(gX**2 + gY**2)
            edgeImg[i-1, j-1] = g
            Gx_img[i-1, j-1] = gX
            Gy_img[i-1, j-1] = gY

    Gx_img_norm = (np.abs(Gx_img) / np.max(np.abs(Gx_img))) * 255
    Gy_img_norm = (np.abs(Gy_img) / np.max(np.abs(Gy_img))) * 255


    edgeImg = np.clip(edgeImg, 0, 255)
    edgeImg[edgeImg < threshold] = 0
    edgeImg[edgeImg >= threshold] = 255
    edgeImg = edgeImg.astype(np.uint8)
    return edgeImg, Gx_img_norm, Gy_img_norm

img = cv2.imread("./bird.jpeg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


edges, gx, gy = edgeDetectionSobel(gray, threshold=100)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.title("Sobel X (Vertikal)")
plt.imshow(gx, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Sobel Y (Horizontal)")
plt.imshow(gy, cmap='gray')
plt.axis('off')
plt.show()


plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.title("Gambar Asli")
plt.imshow(gray, cmap='gray')
plt.axis('off')

plt.subplot(1,2,2)
plt.title("Deteksi Tepi Metode Sobel")
plt.imshow(edges, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()

