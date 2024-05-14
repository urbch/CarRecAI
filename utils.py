import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input


def load_and_preprocess_image(image_path):
    """
    Загружает изображение, изменяет его размер и
    нормализует значения пикселей.
    """
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


def decode_prediction(prediction, class_indices):
    """
    Декодирует предсказание модели в название марки автомобиля.
    """
    predicted_class_index = np.argmax(prediction)
    predicted_class = class_indices[predicted_class_index]
    return predicted_class


def print_prediction(prediction, class_indices):
    """
    Выводит предсказание модели.
    """
    predicted_class = decode_prediction(prediction, class_indices)
    print(f"Предсказанная марка: {predicted_class}")