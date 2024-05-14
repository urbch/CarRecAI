from tensorflow.keras.models import load_model

from utils import load_and_preprocess_image, decode_prediction, print_prediction
import json

# Загрузка модели
model = load_model('car_brand_model.h5')

# Загрузка class_indices из файла
with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)


# Загрузка изображения
image_path = "test_image.jpg"  # Замените на путь к вашему изображению
image = load_and_preprocess_image(image_path)

# Предсказание
prediction = model.predict(image)

# Вывод результата
prediction = model.predict(image)
print(prediction)
