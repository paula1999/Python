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

Además, se le puede añadir ruido adicional antes de que pase a la siguiente iteración para estaibilizar la red neuronal y no converja al mismo elemento.



## Red neuronal

La red neuronal de los modelos de difusión tiene la arquitectura UNet.

![image](https://github.com/paula1999/Python/assets/32401901/03b000b4-b8fa-4c70-84c1-79c1dce26dde)

Puede recibir más información en forma de incrustaciones (embeddings).

- Time embedding: relacionado con el tiempo de cada iteración y el nivel de ruido.
- Context embedding: relacionado con controlar lo que genera el modelo.

![image](https://github.com/paula1999/Python/assets/32401901/9a9eb646-7df4-4b32-8eb9-4402b12ccd3c)

```py
cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1, 1)
temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)

up2 = self.up1(cemb1 * up1 + temb1, down2)
```

La red neuronal aprende a predecir ruido, aunque realmente aprende la distribución de lo que no es ruido.


## Training

A la hora de entrenar una red neuronal, elegimos muestras y le añadimos un nivel de ruido a cada imagen en cada paso.


## Controlling


Embeddings son vectores que captan el significado del texto.

Los textos que tienen contenido similar tienen vectores similares.

De esta manera, se entrena la red neuronal con imágenes junto al texto que las describe, y así se podrá predecir una nueva imagen a partir de un texto.

El contexto es un vector para controlar la generación del modelo, puede ser un embedding de texto o categorías.


## Fast Sampling (DDIM)

Con DDIM (Denoising Diffusion Implicit Models) se puede acelerar el sampling.

DDIM se salta algunos timesteps, predice una idea de la salida final y la define mediante el proceso de eliminación de ruido.
