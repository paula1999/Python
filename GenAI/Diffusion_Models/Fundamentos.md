# Diffusion Models

La red neuronal aprende a eliminar el ruido que se le añade a las imágenes.

Cuando hay mucho ruido, este sigue una distribución normal.

Cuando se le pidea la red neuronal un nuevo sprite:

- Se puede muestrear el ruido de la distribución normal.
- Conseguir un nuevo sprite usando la red para eliminar el ruido.

De esta forma, se pueden generar nuevos sprites a partir de los datos de entrenamiento.


## Sampling

La red neuronal intenta predecir todo el ruido en cada paso.

Primero, se le pasa como entrada a la red neuronal entrenada una muestra de ruido. Esta red neuronal predice el ruido y elimina dicho ruido de la muestra de entrada. Esto se repite multiples veces para conseguir sprites de alta calidad.
