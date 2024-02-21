# CNN
Problema: Clasificación de imágenes de frutas en un conjunto de datos de entrenamiento.

Conjunto de datos con imágenes de frutas (manzanas, plátanos, naranjas).
Etiquetas correspondientes a cada imagen indicando el tipo de fruta.

```py
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator # Procesar imágenes

# Cargar y preprocesar el conjunto de datos
# Supongamos que tienes un directorio llamado "frutas" con subdirectorios "manzanas", "platanos", "naranjas".
train_datagen = ImageDataGenerator(rescale=1./255) # Crear generador de datos de imágenes y normalizar los valores de píxeles de las imágenes entre 0 y 1
train_generator = train_datagen.flow_from_directory(
    'ruta/del/directorio/frutas',
    target_size=(100, 100), # Tamaño de las imágenes
    batch_size=32, # Número de imágenes que se procesan a la vez
    class_mode='categorical' # Problema de clasificación categórica
)

# Construir el modelo CNN
model = models.Sequential() # Modelo secuencial, una pila lineal de capas

# Operaciones de covolución en las imágenes de entrada
# Filtros (32 o 64): número de patrones diferentes que se buscan en las imágenes
# Tamaño del kernel (3, 3): ventana o matriz que se desplaza sobre la imagen para realizar la convolución
# Función de activación (relu): para introducir no linealidades en la red
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))

# Reduce la dimensionalidad espacial de la representación y el número de parámetros en la reed, disminuyendo el costo computacional
# Tamaño del pool (2, 2): ventana sobre la cual se tomará el máximo valor
model.add(layers.MaxPooling2D((2, 2)))


model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

# Transformar los datos en un vector unidimensional.
# Es necesario antes de conectar las capas densas para aplanar la salida de las capas convolucionales y de agrupación
model.add(layers.Flatten())

# Operaciones de conexión total entre las neuronas, aplicando transformaciones lineales
# Número de neuronas (64): unidades tiene la capa
# Función de activación (relu): no linealidades en la red
model.add(layers.Dense(64, activation='relu'))

# Salida final de la red
# Número de neuronas (3): clasificación de 3 clases
# Función de activación (softmax): convertir las salidas en una distribución de probabilidad
model.add(layers.Dense(3, activation='softmax'))

# Compilar el modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
model.fit(train_generator, epochs=10)

# Evaluar el modelo en un conjunto de datos de prueba (no proporcionado en este ejemplo)
model.evaluate(test_generator)

# Utilizar el modelo entrenado para clasificar nuevas imágenes
prediction = model.predict(nueva_imagen_preprocesada)
```
