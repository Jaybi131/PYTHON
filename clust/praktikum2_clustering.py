
#Этот код импортирует библиотеки OpenCV (cv2) и NumPy (np) для обработки изображений и работы с многомерными массивами соответственно,
# а также определяет путь к изображению 'image_fish.jpg'.
import cv2
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
import os
# Путь к изображению
image_path = 'image_fish.jpg'
print(image_path)


def load_image(image_path):
    image = cv2.imread(image_path)
     # Загрузка изображения
    image = cv2.imread(image_path)
    
    # Конвертация из BGR в RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Преобразование изображения в список пикселей [(Высота * Ширина), 3]
    pixels = image_rgb.reshape((-1, 3))
    
    # Конвертация пикселей в float32
    pixels = np.float32(pixels)
    
    return image_rgb, pixels

# Загрузка изображения и преобразование пикселей
image, pixels = load_image(image_path)

# Проверка результата
print(image.shape)  # Ожидается: (Высота, Ширина, 3)
print(pixels.shape) # Ожидается: (Высота * Ширина, 3)
print(pixels.dtype) # Ожидается: float32

image, pixels = load_image(image_path)

def kmeans_clustering(pixels, k, max_iter, n_init):
    # Определение критериев остановки для k-means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, 0.2)
    
    # Применение k-means кластеризации
    compactness, labels, centers = cv2.kmeans(pixels, k, None, criteria, n_init, cv2.KMEANS_RANDOM_CENTERS)
    
    # Конвертация центров кластеров в целочисленный тип
    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape((image.shape))
   # return centers, labels
    
    return segmented_image, centers, labels
    
def calculate_inertia(pixels, labels, centers):
    inertia = 0
    for i, center in enumerate(centers):
        cluster_points = pixels[labels.flatten() == i]  # Используем flatten() для одномерного массива labels
        distances = np.linalg.norm(cluster_points - center, axis=1)
        inertia += np.sum(distances**2)
    return inertia
    
def grid_search_kmeans(pixels, k_values, max_iter_values, n_init_values):
    best_inertia = float('inf')
    best_params = None
    results = []

    for k, max_iter, n_init in product(k_values, max_iter_values, n_init_values):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, 0.2)
        compactness, labels, centers = cv2.kmeans(pixels, k, None, criteria, n_init, cv2.KMEANS_RANDOM_CENTERS)
        
        inertia = calculate_inertia(pixels, labels, centers)
        results.append((k, max_iter, n_init, inertia))
        
        if inertia < best_inertia:
            best_inertia = inertia
            best_params = (k, max_iter, n_init)
   
    
    return best_params, best_inertia, results


k_values = range(2, 11)
max_iter_values = [100, 200, 300]
n_init_values = [10, 20, 30]
best_params, best_inertia, results = grid_search_kmeans(pixels, k_values, max_iter_values, n_init_values)
print(f'Grid Search: Best params={best_params} with inertia={best_inertia}')

def plot_elbow_method(results):
    k_values = sorted(set(result[0] for result in results))
    inertia_values = [min(result[3] for result in results if result[0] == k) for k in k_values]
    
    plt.figure(figsize=(8, 6))
    plt.plot(k_values, inertia_values, marker='o')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    plt.grid(True)
    plt.show()

plot_elbow_method(results)

def quantize_image(pixels, labels, centers):
    """
    Квантовать изображение, заменяя каждый пиксель на цвет ближайшего центра кластера.

    :param pixels: исходное изображение в формате numpy массива
    :param labels: метки кластеров для каждого пикселя
    :param centers: центры кластеров
    :return: квантованное изображение
    """
    # Создаем пустое изображение для квантованных пикселей
    quantized_image = centers[labels.flatten()]
    
    # Восстанавливаем форму изображения
    quantized_image = quantized_image.reshape(image.shape)
    
    return quantized_image

# Кластеризация изображения с лучшими параметрами

segmented_image, centers, labels = kmeans_clustering(pixels, best_params[0], best_params[1], best_params[2])
# Квантование изображения
quantized_image = quantize_image(pixels, labels, centers)

# Сохранение квантованного изображения
quantized_image_path = 'quantized_image.jpg'
cv2.imwrite(quantized_image_path, cv2.cvtColor(quantized_image, cv2.COLOR_RGB2BGR))

def calculate_storage_size(image_path, quantized_image_path):
    original_size = os.path.getsize(image_path)
    quantized_size = os.path.getsize(quantized_image_path)
    
    return original_size, quantized_size
image_path = 'image_fish.jpg'  # Убедитесь, что этот путь соответствует пути к вашему изображению
quantized_image_path = 'quantized_image.jpg'
original_size, quantized_size = calculate_storage_size(image_path, quantized_image_path)
print(f'Original image size: {original_size / 1024:.2f} KB')
print(f'Quantized image size: {quantized_size / 1024:.2f} KB')

def plot_results(original_image, segmented_image, quantized_image, centers, best_params, original_size, quantized_size):
    plt.figure(figsize=(15, 5))

    plt.figure(figsize=(15, 5))

    # Оригинальное изображение
    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(original_image)
    plt.axis('off')

    # Сегментированное изображение
    plt.subplot(1, 3, 2)
    plt.title('Segmented Image')
    plt.imshow(segmented_image)
    plt.axis('off')

    # Квантованное изображение
    plt.subplot(1, 3, 3)
    plt.title('Quantized Image')
    plt.imshow(quantized_image)
    plt.axis('off')

    plt.show()

    # Display the dominant colors
    plt.figure(figsize=(8, 2))
    for i, color in enumerate(centers):
        plt.subplot(1, len(centers), i + 1)
        plt.axis('off')
        plt.imshow(np.ones((100, 100, 3), dtype='uint8') * color)
    plt.show()

plot_results(image, segmented_image, quantized_image, centers, best_params, original_size, quantized_size)


 

 


