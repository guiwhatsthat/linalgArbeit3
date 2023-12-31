import numpy as np
from scipy.fftpack import dct, idct
from PIL import Image
import matplotlib.pyplot as plt

def load_image_from_file(file_path):
    """
    Lädt ein Bild von einer Datei und konvertiert es in ein numpy-Array.
    """
    image = Image.open(file_path).convert('L')  # Konvertiert das Bild zu Graustufen
    return np.array(image)

def dct_2d(image):
    """
    Führt die 2D-DCT-Transformation auf ein Bild an.
    """
    dct_temp = dct(image, axis=0, norm='ortho')
    dct_image = dct(dct_temp, axis=1, norm='ortho')
    return dct_image

def idct_2d(dct_image):
    """
    Führt die inverse 2D-DCT-Transformation auf ein DCT-Bild an.
    """
    idct_temp = idct(dct_image, axis=1, norm='ortho')
    idct_image = idct(idct_temp, axis=0, norm='ortho')
    return idct_image

def save_image(image, file_path):
    """
    Speichert ein Bild in einem gegebenen Dateipfad.
    """
    plt.imsave(file_path, image, cmap='gray')

def plot_images(original, transformed, reconstructed):
    """
    Zeigt das Originalbild, das transformierte (mit logarithmischer Skalierung) Bild und das rekonstruierte Bild nebeneinander an.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title('Originalbild')
    axes[0].axis('off')

    # Logarithmische Skalierung für das transformierte Bild
    axes[1].imshow(np.log(np.abs(transformed) + 1), cmap='gray')
    axes[1].set_title('DCT-transformiertes Bild (log. Skalierung)')
    axes[1].axis('off')

    axes[2].imshow(reconstructed, cmap='gray')
    axes[2].set_title('Rekonstruiertes Bild')
    axes[2].axis('off')

    plt.show()

# Pfade zu Ihren lokalen Bildern
linienbild_path = "BildPath"
normalesbild_path ="BildPath"

# Laden und Verarbeiten des Linienbildes
linienbild = load_image_from_file(linienbild_path)
dct_linienbild = dct_2d(linienbild)
reconstructed_linienbild = idct_2d(dct_linienbild)

# Speichern des DCT- und IDCT-Bildes für das Linienbild
save_image(dct_linienbild, 'dct_linienbild.jpg')
save_image(reconstructed_linienbild, 'reconstructed_linienbild.jpg')

# Laden und Verarbeiten des normalen Bildes
normalesbild = load_image_from_file(normalesbild_path)
dct_normalesbild = dct_2d(normalesbild)
reconstructed_normalesbild = idct_2d(dct_normalesbild)

# Speichern des DCT- und IDCT-Bildes für das normale Bild
save_image(dct_normalesbild, 'dct_normalesbild.jpg')
save_image(reconstructed_normalesbild, 'reconstructed_normalesbild.jpg')

# Anzeigen der Bilder für das Linienbild
plot_images(linienbild, dct_linienbild, reconstructed_linienbild)

# Anzeigen der Bilder für das normale Bild
plot_images(normalesbild, dct_normalesbild, reconstructed_normalesbild)
