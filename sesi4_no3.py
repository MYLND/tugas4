import imageio
import numpy as np
import matplotlib.pyplot as plt

# Membaca citra dengan mode grayscale
image = imageio.imread('sesi4.jpeg', mode='F')
image = image / 255.0  # Pastikan nilai piksel dalam rentang 0-1

# Menghitung histogram dan histogram kumulatif
histogram, bin_edges = np.histogram(image, bins=256, range=(0, 1))
cdf = histogram.cumsum()  # histogram kumulatif
cdf_normalized = cdf / cdf.max()  # normalisasi cdf ke rentang 0-1

# Melakukan ekualisasi histogram
image_equalized = np.interp(image.flatten(), bin_edges[:-1], cdf_normalized)
image_equalized = image_equalized.reshape(image.shape)

# Menampilkan hasil
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.imshow(image, cmap='gray')
ax1.set_title('Citra Asli')
ax2.imshow(image_equalized, cmap='gray')
ax2.set_title('Citra Setelah Ekualisasi Histogram')
plt.show()
