import numpy as np
import matplotlib.pyplot as plt

def create_dct_matrix(N):
    """
    Erzeugt eine NxN DCT Transformationsmatrix.
    """
    # Initialisiere die NxN-Matrix
    dct_matrix = np.zeros((N, N))
    for k in range(N):
        for n in range(N):
            if k == 0:
                dct_matrix[k, n] = np.sqrt(1/N)
            else:
                dct_matrix[k, n] = np.sqrt(2/N) * np.cos((np.pi * (2*n + 1) * k) / (2 * N))
    return dct_matrix

def dct_transform(vector, dct_matrix):
    """
    Führt die DCT-Transformation auf einen Vektor mit der gegebenen DCT-Matrix aus.
    """
    return np.dot(dct_matrix, vector)

def idct_transform(vector, dct_matrix):
    """
    Führt die inverse DCT-Transformation auf einen Vektor mit der gegebenen DCT-Matrix aus.
    """
    return np.dot(dct_matrix.T, vector)

# Demonstration der Funktionen
N = 8  # Grösse der DCT-Matrix
example_vector = np.random.rand(N)  # Erzeuge einen zufälligen Vektor für das Beispiel
print(example_vector)
# Erzeuge die DCT-Matrix
dct_matrix = create_dct_matrix(N)

# Führt die DCT-Transformation durch
transformed_vector = dct_transform(example_vector, dct_matrix)


# Führt die inverse DCT-Transformation durch
reconstructed_vector = idct_transform(transformed_vector, dct_matrix)
print(reconstructed_vector)

# Visualisierung
plt.figure(figsize=(12, 8))

# Originaler Vektor
plt.subplot(3, 1, 1)
plt.stem(example_vector, use_line_collection=True)
plt.title('Originaler Vektor')
plt.xlabel('Index')
plt.ylabel('Wert')

# DCT-transformierter Vektor
plt.subplot(3, 1, 2)
plt.stem(transformed_vector, use_line_collection=True)
plt.title('DCT-transformierter Vektor')
plt.xlabel('Index')
plt.ylabel('Wert')

# Rekonstruierter Vektor
plt.subplot(3, 1, 3)
plt.stem(reconstructed_vector, use_line_collection=True)
plt.title('Rekonstruierter Vektor nach IDCT')
plt.xlabel('Index')
plt.ylabel('Wert')

plt.tight_layout()
plt.show()