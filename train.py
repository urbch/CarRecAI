import os
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import create_model


# Параметры модели
num_classes = 5
image_height = 224
image_width = 224
batch_size = 32
epochs = 10

# Пути к данным
train_data_dir = "data/train"
validation_data_dir = "data/validation"

# Создание модели
model = create_model(num_classes, image_height, image_width)
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Генераторы данных
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(image_height, image_width),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = val_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(image_height, image_width),
    batch_size=batch_size,
    class_mode='categorical')

# Обучение модели
model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator)

# Сохранение модели
model.save('car_brand_model.h5')

# Сохранение class_indices в файл
import json
with open('class_indices.json', 'w') as f:
    json.dump(train_generator.class_indices, f)

# # Fine-tuning (опционально)
# for layer in model.layers[-10:]:
#     layer.trainable = True
#
# model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
#
# model.fit(
#     train_generator,
#     epochs=5,
#     validation_data=validation_generator)
#
# model.save('car_brand_model_finetuned.h5')
