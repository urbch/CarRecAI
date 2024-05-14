from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model


def create_model(num_classes, image_height, image_width):
    # Загрузка ResNet50 без верхних слоёв
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(image_height, image_width, 3))

    # Замораживание весов базовой модели
    base_model.trainable = False

    # Добавление полносвязных слоёв
    x = base_model.output
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    # Создание модели
    model = Model(inputs=base_model.input, outputs=predictions)
    return model
