# Generative Adversarial Networks (GANs)

Un GAN está formado por un generador y un discriminador.

En un GAN básico, el generador toma como entrada un ruido aleatorio y produce ejemplos falsos. Estos ejemplos junto a ejemplos reales se le pasan al discriminador, que devuelve una probabilidad que indica si es real o falso. El objetivo del discriminador es distinguir entre los ejemplos generados y los reales, mientras que el objetivo del generador es engañar al discriminador produciendo ejemplos falsos que parezcan lo más reales posibles.

Para entrenar un GAN, se alterna entre el entrenamiento del generador y del discriminador.

El discriminador, después de devolver la probabilidad, compara las predicciones usando una función de pérdida BCE con las etiquetas de real o falso. De esta forma, se actualizan los parámetros del discriminador

El generador primero produce algunos ejemplos falsos y se los pasa al discriminador. En este caso, el generador solo puede ver sus propios ejemplos falsos y la salida del discriminador indicando la probabilidad. Después de esta predicción, se compara usando la función de pérdida BCE siendo las etiquetas de estos ejemplos como reales. Una vez calculado el coste, el gradiente se propaga hacia atrás y los parámetros del generator se actualizan.


## Funciones de activación

- Son no lineales y diferenciables.
- Diferenciables para la propagación hacia atrás.
- No lineales para aproximarse a funciones complejas.

### ReLU

Toma el máximo entre cada valor y 0, por lo que básicamente transforma los números negativos en 0.

### Leaky ReLU

Al igual que ReLU, los números positivos no cambian. Sin embargo, los números negativos se multiplican por una constante (normalmente 0.1).

### Sigmoid

Cuando el valor es positivo, devuelve un número entre 0.5 y 1, mientras que cuando el valor es negativo, devuelve un número entre 0 y 0.5. Por lo que siempre devuelve un número entre 0 y 1, se suele usar en modelos de clasificación binaria para indicar una probabilidad entre 0 y 1.

No se suele usar en capas ocultas porque en los extremos las derivadas son casi nulas.

### Tanh

Devuelve valores entre -1 y 1. Cuando el valor es positivo, devuelve un número entre 0 y 1; mientras que cuando el valor es negativo, devuelve un número entre -1 y 0. De esta forma, conserva el signo de cada valor.


## Batch Normalization

Durante el entrenamiento, en cada capa se normalizan los valores para que tengan media 0 y desviación estándar 1, y luego se reescalan con dos parámetros: *shift factor* y *scale factor*.

Durante el test, se usa la media y desviación estándar del conjunto de entrenamiento.


## Convolutions

Las convoluciones nos permiten detectar características claves en diferentes áreas (ventanas) de una imagen usando filtros.

![image](https://github.com/paula1999/Python/assets/32401901/789b967a-ec1a-4048-a540-04d962990fc2)

### Stride

Indica cada cuantos bloques a la derecha o hacia abajo se aplica el filtro.

### Padding

Normalmente, el filtro se aplica más veces en los píxeles del centro que en los extremos. Si queremos darle importancia a los extremos, se puede crear un marco sobre la imagen de tal forma que toda la información quede en el centro de la imagen y cada pixel se visita el mismo número de veces.

## Pooling

Se usa para reducir el tamaño de la entrada.

- **Max pooling**: toma el máximo de entre los valores de cada pixel en cada ventana. Sirve para extraer la información destacada.
- **Average pooling**.
- **Min pooling**.


## Upsampling

Se usa para aumentar el tamaño de la entrada, es decir, devolver una imagen con mayor resolución. Esto se hace infiriendo valores para los píxeles adicionales.

- **Nearest Neighbors**. Primero, se asigna el valor de la esquina de arriba a la izquierda igual al valor de entrada de arriba a la izquierda. El resto de valores de los píxeles de entrada se añaden con una distancia de ciertos píxeles, dependiendo del tamaño de salida. Finalmente, se asigna el mismo valor a los píxeles cercanos.

| a | b |
|---|---|
| c | d |   

| a | a | b | b |
|---|---|---|---|
| a | a | b | b |
| c | c | d | d |
| c | c | d | d |

- **Linear interpolation**.
- **Bi-linear interpolation**.


## Transposed Convolutions

Son convolutiones con upsampling y tienen parámetros aprendibles.

![image](https://github.com/paula1999/Python/assets/32401901/380ce683-8005-4c93-9147-24786ab5fcde)


## Conditional Generation

Permite especificar la clase que el generador tiene que producir.


| Condicional                                        | No condicional                                             |
| ---                                                | ---                                                        |
| Ejemplos de clases que quieres                     | Ejemplos de clases aleatorias                              |
| El conjunto de entrenamiento debe estar etiquetado | El conjunto de entrenamiento no tiene que estar etiquetado |

- La clase se le pasa al generador como un one-hot vector.
- La clase se le pasa al discriminador como una one-hot matriz.
- El tamaño del vector y el número de matrices representan el número de clases.


