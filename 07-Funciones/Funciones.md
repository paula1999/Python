# 7. Funciones
## 7.1 Funciones
Tienen el siguiente formato:
```py
def nombre_de_la_funcion (parametro1, ..., parametroN):
    instrucciones

    return variable # No es obligatorio
```

## 7.2 Funciones lambda
A veces las funciones solamente requieren dos o tres líneas para su realización. En esos casos Python tiene otra forma de declarar las funciones en una sola línea. Dichas funciones se conocen como funciones **lambda** y su formato es el siguiente:

```py
f = lambda parámetros: expresión de la función f
```

Por ejemplo, la función `f(x) = 3*x` es:

```py
f = lambda x: 3*x
```

que se puede usar como

```py
>>> f(8)
24
```

## 7.3 Llamado por valor y por referencia
Todos los vectores, listas, matrices... se pasan por referencia (se pueden modificar en la función).

## 7.4 Variables locales y globales
Las variables que corresponden al mismo parámetro en el programa principal y en la función NO se comparten. Las variables correspondientes a la función solamente son accesibles por la función. Para hacer que una variable sea **global** hay que definirla de la siguiente manera: `global variable`.
