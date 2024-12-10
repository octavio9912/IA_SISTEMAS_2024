import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import os

# Deshabilitar opciones de optimización no deseadas
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Función para preprocesar imágenes
def preprocess_image(image, target_size=(64, 64)):
    if isinstance(image, str):  
        image = cv2.imread(image)
        if image is None:
            raise FileNotFoundError(f"No se pudo leer la imagen en la ruta: {image}")
    elif not isinstance(image, np.ndarray):
        raise ValueError("El formato de la imagen no es compatible. Debe ser una ruta o un array de NumPy.")

    try:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, target_size)
    except Exception as e:
        raise ValueError(f"Error al preprocesar la imagen: {e}")

    return image / 255.0  

# Cargar y procesar el conjunto de datos
def load_dataset(image_paths, labels, target_size=(64, 64)):
    images = []
    for img_path in image_paths:
        try:
            images.append(preprocess_image(img_path, target_size))
        except Exception as e:
            print(f"Advertencia: {e}")
            continue
    return np.array(images), np.array(labels)

# Definición de rutas y etiquetas de imágenes
image_paths = [
      'C:/Users/octav/Downloads/Examen/Examen/imagenes/melanoma1.jpg',
    'C:/Users/octav/Downloads/Examen/Examen/imagenes/melanoma2.jpg',
    'C:/Users/octav/Downloads/Examen/Examen/imagenes/melanoma3.jpg',
    'C:/Users/octav/Downloads/Examen/Examen/imagenes/melanoma4.jpg',
    'C:/Users/octav/Downloads/Examen/Examen/imagenes/melanoma5.jpg',
    'C:/Users/octav/Downloads/Examen/Examen/imagenes/melanoma6.jpg',
    'C:/Users/octav/Downloads/Examen/Examen/imagenes/melanoma7.jpg',
    'C:/Users/octav/Downloads/Examen/Examen/imagenes/melanoma8.jpg',
    'C:/Users/octav/Downloads/Examen/Examen/imagenes/melanoma9.jpg',
    'C:/Users/octav/Downloads/Examen/Examen/imagenes/melanoma10.jpg',
    'C:/Users/octav/Downloads/Examen/Examen/imagenes/eczema1.jpg',
    'C:/Users/octav/Downloads/Examen/Examen/imagenes/eczema2.jpg',
    'C:/Users/octav/Downloads/Examen/Examen/imagenes/eczema3.jpg',
    'C:/Users/octav/Downloads/Examen/Examen/imagenes/eczema4.jpg',
    'C:/Users/octav/Downloads/Examen/Examen/imagenes/eczema5.jpg',
    'C:/Users/octav/Downloads/Examen/Examen/imagenes/eczema6.jpg',
    'C:/Users/octav/Downloads/Examen/Examen/imagenes/eczema7.jpg',
    'C:/Users/octav/Downloads/Examen/Examen/imagenes/eczema8.jpg',
    'C:/Users/octav/Downloads/Examen/Examen/imagenes/eczema9.jpg',
    'C:/Users/octav/Downloads/Examen/Examen/imagenes/eczema10.jpg',
    'C:/Users/octav/Downloads/Examen/Examen/imagenes/eczema11.jpg',
]
labels = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  #  0  Melanoma, 1  Eczema

# Cargar imágenes y etiquetas
X, y = load_dataset(image_paths, labels)
if len(X) == 0:
    raise ValueError("No se pudieron cargar imágenes. Verifique las rutas y los formatos.")

# Codificar etiquetas en formato one-hot
y = to_categorical(y, num_classes=2)

# Dividir en conjuntos de entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Construcción del modelo
model = Sequential([
    Flatten(input_shape=(64, 64, 3)),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(2, activation='softmax')  
])

# Compilación del modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Configuración de EarlyStopping para detener entrenamiento temprano si no mejora
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Entrenamiento del modelo
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=16, callbacks=[early_stopping])

# Guardar el modelo entrenado
model.save('skin_disease_model.h5')

# Función para realizar predicciones
def predict_disease(image, model, target_size=(64, 64), confidence_threshold=0.6):
    img = preprocess_image(image, target_size)
    img = np.expand_dims(img, axis=0)  
    predictions = model.predict(img)
    max_confidence = np.max(predictions)
    class_idx = np.argmax(predictions)
    if max_confidence < confidence_threshold:
        return "Sin coincidencias", max_confidence
    return ["Melanoma", "Eczema"][class_idx], max_confidence

# Captura de video y detección en tiempo real
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("No se pudo acceder a la cámara. Verifique la conexión.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo capturar el cuadro de video.")
            break

        # Clasificar enfermedad en la piel
        label, confidence = predict_disease(frame, model)
        text = f"{label} ({confidence*100:.1f}%)" if label != "Sin coincidencias" else label

        # Mostrar resultado en la ventana
        cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Detección de Enfermedades de Piel', frame)

        # Salir con la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()




  
