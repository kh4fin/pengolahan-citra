import cv2
import numpy as np
import matplotlib.pyplot as plt

# Baca gambar dan konversi ke grayscale
img = cv2.imread('./bird.jpeg', cv2.IMREAD_GRAYSCALE)

# Deteksi tepi menggunakan Canny
edges = cv2.Canny(img, 100, 200)

def find_chain_codes(binary_img):
    visited = np.zeros_like(binary_img, dtype=bool)
    h, w = binary_img.shape
    contours_with_chain = []

    # 8 arah dan mapping kode chain-nya
    directions = [(-1, 1), (0, 1), (1, 1),
                  (1, 0), (1, -1), (0, -1),
                  (-1, -1), (-1, 0)]  # arah sesuai chain code 0–7

    def trace_chain(y, x):
        start = (y, x)
        chain_code = []
        visited[y, x] = True

        while True:
            found = False
            for idx, (dy, dx) in enumerate(directions):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:     # Cek apakah tetangga masih berada dalam batas gambar
                    if binary_img[ny, nx] == 255 and not visited[ny, nx]:       # Cek apakah tetangga tersebut adalah bagian dari tepi (255)
                        chain_code.append(idx)
                        visited[ny, nx] = True
                        y, x = ny, nx
                        found = True
                        break
            if not found:
                break           # ika tidak ditemukan tetangga yang valid di 8 arah, maka kontur sudah selesai dilacak

        return chain_code

    for i in range(h):
        for j in range(w):
            if binary_img[i, j] == 255 and not visited[i, j]:       # Cek apakah piksel adalah bagian dari tepi (255)
                chain = trace_chain(i, j)
                if len(chain) > 10:  # abaikan noise kecil
                    contours_with_chain.append(chain)
    return contours_with_chain

# Jalankan deteksi kontur dan simpan Chain Code-nya
chain_codes = find_chain_codes(edges)

# Visualisasi: gambar kontur seperti sebelumnya
output_img = np.ones((*edges.shape, 3), dtype=np.uint8) * 255  # putih

for i in range(edges.shape[0]):
    for j in range(edges.shape[1]):
        if edges[i, j] == 255:
            output_img[i, j] = [0, 0, 0]

# Tampilkan hasil
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Edge Deteksi")
plt.imshow(edges, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Visual Kontur")
plt.imshow(output_img)
plt.axis('off')

plt.tight_layout()
plt.show()

# Cetak contoh Chain Code kontur
print("Contoh Chain Code untuk kontur pertama (maks 50 arah):")
print(chain_codes[0][:50] if chain_codes else "Tidak ada kontur ditemukan.")
