import cv2
import numpy as np
import matplotlib.pyplot as plt


# ========== FUNGSI: Prewitt ==========
def edgeDetectionPrewitt(img, threshold=100):
    pX = np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ], dtype=np.float32)

    pY = np.array([
        [-1, -1, -1],
        [ 0,  0,  0],
        [ 1,  1,  1]
    ], dtype=np.float32)

    h, w = img.shape
    edgeImg = np.zeros((h-2, w-2), dtype=np.float32)
    Gx_img = np.zeros((h-2, w-2), dtype=np.float32)
    Gy_img = np.zeros((h-2, w-2), dtype=np.float32)

    for i in range(1, h-1):
        for j in range(1, w-1):
            region = img[i-1:i+2, j-1:j+2]
            gX = np.sum(region * pX)
            gY = np.sum(region * pY)
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


img = cv2.imread("bird.jpeg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

edges_prewitt, gx_prewitt, gy_prewitt = edgeDetectionPrewitt(gray, threshold=100)


# ========== VISUALISASI ==========
plt.figure(figsize=(14, 8))

plt.subplot(2, 3, 1)
plt.title("Gambar Asli (Grayscale)")
plt.imshow(gray, cmap='gray')
plt.axis('off')


plt.subplot(2, 3, 3)
plt.title("Tepi Prewitt")
plt.imshow(edges_prewitt, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
