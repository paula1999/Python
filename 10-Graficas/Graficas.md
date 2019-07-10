# 10. Gráficas
## 10.1 Visualización de datos
Usaremos las bibliotecas de graficación del paquete `matplotlib`, como por ejemplo la biblioteca `pyplot`. Por lo que siempre que queramos graficar tenemos que importar esta biblioteca.

## 10.2 Gráficas en 2 dimensiones
La instrucción es `plot(y)`, donde `y` es una lista de datos. Esta instrucción se carga con: `import matplotlib.pyplot`. Para desplegar la gráfica usaremos `show()`. Por lo que si solo queremos cargar estas dos instrucciones haremos: `from matplotlib.pyplot import plot, show`.

Lo mismo sirve cuando queremos graficas una list (x, y).

Si queremos cambiar los límites de los ejes usaremos las instrucciones `xlim` e `ylim` de la siguiente manera:

```py
xlim(límite inferior de x, límite superior de x)
ylim(límite inferior de y, límite superior de y)
```

Podemos generar la lista de valores para los ejes `x`, `y` usando la instrucción `linspace` de la biblioteca `numpy`:

```py
linspace(límite inferior, límite superior, número de puntos)
```

Si queremos cambiar las **etiquetas de los ejes**, por ejemplo en el eje `x` queremos que se muestren en términos de pi, por lo que haremos:
```py
from matplotlib.pyplot import xticks

xticks([-pi, -pi/2, pi/2, pi], [r'$-\pi$', 'r$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
```

También podemos añadir una **leyenda** usando las instrucciones `label` (se añade dentro de la instrucción `plot`) y `legend`. Por ejemplo:

```py
# Queremos realizar las gráficas de `sen(x)` y `cos(x)` añadiendo la leyenda `sen(x)` y `cos(x)`
from numpy import linspace, pi, sin, cos
from matplotlib.pyplot import plot, show, xlim, ylim, legend, xticks

x = linspace(-pi, pi, 256)
y = sin(x)
z = cos(x)

plot(x, y, label = 'seno')
plot(x, z, label = 'coseno')
legend(loc = 'upper left')
xticks ([-pi, -pi/2, pi/2, pi], [r'$-\pi$', 'r$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
ylim(-2, 2)
show()
```

## 10.3 Figuras múltiples
Para poder realizar varias figuras es necesario numerarlas con la instrucción `figure`, por ejemplo:

```py
a = figure(2)
```

Se debe importar `figure` de `matplotlib`.

Por ejemplo:

```py
# Queremos realizar las gráficas de `sen(x)` y `cos(x)` añadiendo la leyenda `sen(x)` y `cos(x)`
from numpy import linspace, pi, sin, cos
from matplotlib.pyplot import plot, show, xlim, ylim, legend, xticks, figure

x = linspace(-pi, pi, 256)
y = sin(x)
z = cos(x)
a = figure(2)

plot(x, y, label = 'seno')
legend(loc = 'upper left')

b = figure(20)

plot(x, z, label = 'coseno')
legend(loc = 'upper left')

xticks ([-pi, -pi/2, pi/2, pi], [r'$-\pi$', 'r$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
ylim(-2, 2)
show()
```

## 10.4 Subgráficas
Para realizar varias gráficas en la misma figura usamos subgráficas con `subplot(m, n, k)`, que divide la figura en `m x n` subgráficas en forma de matriz de `m` filas y `n` columnas. La variable `k` numera las subgráficas de izquierda a derecha y de arriba hacia abajo en numeración consecutiva y hace que se active la subgráfica correspondiente. Por ejemplo:

```py
# Queremos realizar las gráficas de `sen(x)` y `cos(x)` añadiendo la leyenda `sen(x)` y `cos(x)`
from numpy import linspace, pi, sin, cos
from matplotlib.pyplot import plot, show, xlim, ylim, legend, xticks, subplot

x = linspace(-pi, pi, 256)
y = sin(x)
z = cos(x)

subplot(2, 1, 1) # n = k = 1, gráfica en la primera fila
plot(x, y, label = 'seno')
legend(loc = 'upper left')

subplot(2, 1, 2) # k = 2, gráfica en la segunda fila
plot(x, z, label = 'coseno')
legend(loc = 'upper left')

xticks ([-pi, -pi/2, pi/2, pi], [r'$-\pi$', 'r$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
ylim(-2, 2)
show()
```

Podemos sustituir `subplot(2, 1, 1)` por `subplot(1, 2, 1)` y `subplot(2, 1, 2)` por `subplot(1, 2, 2)`, y saldrían las gráficas en la misma fila.

## 10.5 Otros tipos de gráficas bidimensionales
### 10.5.1 Gráfica polar
Se obtiene usando `plot` pero usando las componentes `r` (radio vector) y `theta` (ángulo o argumento). Cualquiera de los dos o los dos pueden variar en un rango determinado. Por ejemplo si `r = theta`, podemos variar uno de los dos o los dos. Por ejemplo:

```py
# Cálculo de la gráfica polar de r = 2*pi*theta
from numpy import pi, arange, linspace
from matplotlib.pyplot import subplot, plot, show, grid, title

teta = arange(0, 3, 0.01) # se generan los valores de theta
r = 2*pi*teta

subplot(111, polar = True) # se crea la subgráfica polar
plot (r, teta, color = 'r', linewidth = 3) # se crea la gráfica
grid(True) # se añade la retícula
title("Gráfica polar de r = 2$\pi\\theta$")
show()
```

### 10.5.2 Gráfica de pie
Se usa para representar en porcentaje los datos usando `pie`. Por ejemplo:

```py
# Vamos a graficar los datos [15, 30, 40, 5, 10]
from matplotlib.pyplot import figure, subplot, title, pie, show

figure(1, figsize = (6, 6)) # hace cuadrada la figura
fracs = [15, 30, 40, 5, 10]
pie (fracs, autopct = '%2.1i%%') # escribir los porcentajes en la gráfica
title("Gráfica de pie")
show()
```

### 10.5.3 Gráfica de histograma
Se usa para representar distribuciones de datos, es una gráfica de barras. Se forma con `hist`. Por ejemplo:

```py
# Graficar números aleatorios generados con una distribución normal gaussiana
from matplotlib.pyplot import title, hist, show
from numpy.random import normal

numeros_gaussianos = normal(size = 1000)
hist(numeros_gaussianos)
show ()
```

### 10.5.3 Gráfica de `stem` o de puntos
Se usa para graficar señales en forma de puntos. Los puntos están unidos al eje horizontal. Por ejemplo:

```py
from matplotlib.pyplot import stem, show
from numpy import pi, arange

stem (arange(-pi, pi), '-.')
show()
```

## 10.6 Opciones de gráficas

| Símbolo | Descripción |
|---------|-------------|
| - | línea sólida |
| -- | línea discontinua |
| -. | línea punto y guión|
|: | línea de puntos|
| . | puntos|
| o | círculos |
| ^ | triángulos hacia arriba |
| v | triángulos hacia abajo |
| < | triángulos hacia la izquierda |
| > | triángulos hacia la derecha |
| s | cuadrado |
| + | signo más |
| x | cruz |
| D | diamante|
| d | diamante delgado |
| h | hexágono |
| p |pentágono |
|b | azul|
| g | verde |
| r | rojo |
| c | cian |
| m | magenta |
| y | amarillo |
| k | negro |
| w | blanco |

## 10.7 Gráficas tridimensionales
### 10.7.1 Gráfica de una curva paramétrica
Podemos escribir las ecuaciones en términos de un parámetro. Se usa `plot`, pero previamente hay que indicar que es una figura tridimensional con `Axes3D` que se importa de la biblioteca `mpl_toolkits.mplot3d`. Por ejemplo:

```py
''' Grafica de las ecuaciones:
z = r
x = r * sin(theta)
y = r * cos(theta)
'''

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import title, figure, show, plot, legend
from numpy import linspace, sin, cos, pi

a = figure()
a.gca(projection = '3d')
teta = linspace(-4*pi, 4*pi, 100)
r = linspace(0, pi, 100)
z = r
x = r*sin(teta)
y = r*cos(teta)

plot(x, y, z, label = 'Curva paramétrica')
legend()
show()
```

### 10.7.2 Gráfica de una superficie
Requiere que calculemos una malla con `meshgrid` donde debemos especificar el rango, por ejemplo con

```py
# Graficar la función f(X, Y) = sin(X^2 + Y^2)

from matplotlib import cm
from matplotlib.pyplot import plot, figure, show, title
from numpy import arange, sqrt, sin, meshgrid

fig = figure()
az = fig.gca(projection = '3d')

X = arange(-5, 5, 0.25)
Y = arange(-5, 5, 0.25)
X, Y = meshgrid(X, Y)
Z = sin(sqrt(X**2 + Y**2))
A = ax.plot_surface(X, Y, Z, rstride = 1, cstride = 1, cmap = cm.coolwarm)
show()
```

Si ahora añadimos los parámetros:

```py
rstride = 2, cstride = 2, cmap = cm.coolwarm, linewidth = 0
```

para que quede como:

```py
A = ax.plot_surface(X, Y, Z, rstride = 1, cstride = 1, cmap = cm.coolwarm, linewidth = 0)
```

dentro de la instrucción `plot_surface` obtenemos una figura con las líneas muy delgadas. Finalmente, si añadimos la instrucción:

```py
fig.colorbar(A, shrink = 0.5, aspect = 5)
```

Además, si cambiamos los pasos `rstride` y `cstride` a 2 obtenemos la gráfica con la superficie más tosca y aparece la barra de colores lateral.

### 10.7.3 Gráfica de superficie de alambre (wireframe)
Requiere los mismos pasos que el apartado anterior pero cambiando `surface` por `wireframe`:

```py
from matplotlib import cm
from matplotlib.pyplot import plot, figure, show, title
from numpy import arange, sqrt, sin, meshgrid

fig = figure()
ax = fig.gca(projection= '3d')
X = arange(-5, 5, 0.50)
Y = arange(-5, 5, 0.50)
X, Y = meshgrid(X, Y)
Z = sin(sqrt(X**2 + Y**2))
A = ax.plot_wireframe(X, Y, Z)
show()
```

### 10.7.4 Gráfica de superficie con proyección sobre el plano x, y
Lo haremos graficando la superficie y el contorno lo que se puede hacer con `plot_surface` y `contourf`:

```py
from pylab import arange, sin, sqrt, meshgrid, figure, cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import show

fig = figure()
ax = Axes3D(fig)
X = arange(-4, 4, 0.25)
Y = arange(-4, 4, 0.25)
X, Y = meshgrid(X, Y)
R = sqrt(abs(X**3) + Y**2)
Z = sin(R)

ax.plot_surface(X, Y, Z, rstride = 1, cstride = 1, cmap = cm.hot)
ax.contour(X, Y, Z, zdir = 'z', offset = -2, cmap = cm.hot)
ax.set_zlim(-2, 2)
show()
```
