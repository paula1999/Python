# 5. Vectores

## 5.1 Vectores
En Python no es necesario declarar los vectores al principio del programa. Además, se implementan usando listas, luego pueden contener cualquier tipo de dato. Para inicializar un vector haremos lo siguiente: `vector = [None] * n`, donde `n` es el número de elementos y `None` indica que el vector generado está vacío.

Si queremos inicializarlo con otros valores usaremos: `vector = [a, b, c, ..., ]`.

Para acceder a un elemento del vector utilizaremos los corchetes `[]`.

Además, los vectores se pueden definir usando el **método de comprensión**. El formato es: `[f(x) for x in range(A)]`, que genera un vector de valores de *f(x)* para valores de *x = 0, 1, 2, ..., A-1*, es decir, el vector es `[f(0), f(1), ..., f(A-1)]`. Por ejemplo:

```py
>>> [x**2 for x in range(5)]
[0, 1, 4, 9, 16]
```

El tamaño de un vector se puede modificar añadiendo más elementos utilizando la instrucción `append` (se inserta al final del vector).
