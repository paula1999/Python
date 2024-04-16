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
