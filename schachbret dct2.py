from scipy.fftpack import dct
import numpy as np
import matplotlib.pyplot as plt
# Definieren der Größe des Schachbrettmusters
N = 8

# Erzeugen des Schachbrettmusters
signal_2d = np.zeros((N, N))
signal_2d[1::2, ::2] = 1
signal_2d[::2, 1::2] = 1

# Berechnung der 2D-DCT
dct_2d = dct(dct(signal_2d, axis=0, norm='ortho'), axis=1, norm='ortho')

# Plot des Originalsignals und der 2D-DCT
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Original 2D-Signal
ax1.imshow(signal_2d, cmap='gray', interpolation='nearest')
ax1.set_title('Original 2D Signal')
ax1.set_xticks([])
ax1.set_yticks([])

# 2D DCT
ax2.imshow(np.log(abs(dct_2d)), cmap='gray', interpolation='nearest')
ax2.set_title('2D Discrete Cosine Transform')
ax2.set_xticks([])
ax2.set_yticks([])

plt.show()
