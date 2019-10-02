# 1. Fundamentos.
## 1.1 Variables.
### 1.1.1 Tipos de variables.

- **Entera**: no tienen parte decimal. Por ejemplo `a = 5`.
- **Real**: tiene parte decimal. Por ejemplo `b = 2.5`
- **Alfanumérica**: no tiene un valor numérico pero es una cadena de caracteres alfanuméricos. Por ejemplo `c = "Hola"` o `d = 'Hola'`.
- **Lógica**: solo toma los valores `true` o `false`. Por ejemplo `e = True`.

Para saber el tipo de una variable utilizaremos `type(variable)`.

Python es un lenguaje dinámico porque se puede cambiar el tipo de la variable en tiempo de ejecución (Java y C++ no dejan).

No es necesario usar `;` excepto para separar instrucciones en el mismo renglón.

### 1.1.2 Operaciones.
- **Suma**: `c = b + a`.
- **Resta**: `c = b - a`.
- **Multiplicación**: `c = b * a`.
- **División**: `c = b / a`. Si ponemos `c = b // a`, entonces el resultado es la parte entera de la división.
- **Valor absoluto**: `abs(a)`.
- **Potencia**: `pow(a, b)` o `**`.
- **Residuo**: `a % b`.
- **Cociente y residuo**: `divmod(a, b)`.

#### 1.1.2.1 Funciones de la biblioteca `math`.

| Función | Código | Ejemplo |
|---------|--------|---------|
| Valor absoluto | `abs`, `fabs` | `math.abs(-2)`, `math.fabs(-2)` |
| Función exponencial | `exp` | `math.exp(2)` |
| Pontencia a^b | `pow(a, b)` | `math.pow(2, 3)` |
| Raíz cuadrada | `sqrt` | `math.sqrt(3, 2)` |
| Coseno de un ángulo (rad) | `cos` | `math.cos(0.7)` |
| Seno de un ángulo (rad) | `sin` | `math.sin(0.7)` |
| Tangente de un ánguo (rad) | `tan` | `math.tan(0.7)` |
| Conversión rad a grados | `degree` | `degree(0.7)` |
| pi | `pi` | `math.pi` |
| e | `e` | `math.e` |
| Arco coseno | `acos` | `math.acos(0.7)` |
| Arco seno | `asin` | `math.asin(0.7)` |

Para poder usar estas funciones debemos usar una de las siguientes:

```py
>>> import math
>>> from math import *
```

## 1.2 Operadores.
### 1.2.1 Operadores aritméticos.
Los hemos visto en el apartado anterior [Operaciones](#112-operaciones).

### 1.2.2 Operadores relacionales.
Los utilizaremos para comparar.

- **Mayor que**: `>`.
- **Menor que**: `<`.
- **Mayor o igual que**: `>=`.
- **Menor o igual que**: `<=`.
- **Igual a**: `==`.
- **Distinto de**: `!=`.

### 1.2.3 Operadores lógicos.
En orden de precedencia (más a menos):

- **not**: `not` o `!`.
- **and**: `&` o `and`.
- **or exclusivo**: `^`.
- **or**: `|` u `or`.

### 1.2.4 Operadores de asignación.

| `a = a + b` | `a += b` |
|-------------|----------|
| `a = a - b` | `a -= b` |
| `a = a * b` | `a *= b` |
| `a = a / b` | `a /= b` |
| `a = a % b` | `a %= b` |

## 1.3 Comentarios.
Para comentar una sola línea, utilizaremos el símbolo `#`. Si queremos comentar un párrafo, lo pondremos entre tres apóstrofos `'''`.

## 1.4 Mostrar datos.
Para imprimir una variable `a` utilizaremos `print(a)`. Si queremos imprimir dos variables `a` y `b` entonces utilizaremos `print(a, b)`. Además, si queremos mostrar un texto y dentro de él se imprima una variable `a` haremos `print("TEXTO %3.2f TEXTO" % a)`. El campo `%3.2f` indica que al final de la cadena alfanumérica existe una variable de tipo real que se va a imprimir con dos cifras decimales. El número tres no tiene ninguna relevancia.

| **Especificadores de formato** | Salida mostrada |
|--------------------------------|-----------------|
| `d` o `i` | Entero con signo |
| `u` | Entero sin signo |
| `f` | Decimal de punto flotante |
| `e` | Notación científica |
| `g` | Representación más corta entre `f` y `e` |
| `c` | Caracter |
| `s` | Cadena de caracteres |


## 1.5 Recibir datos.
Utilizaremos `input()`. Por ejemplo:
```py
a = input("Introduce el valor de a")
```

Al utilizar esto, leemos una cadena. Si queremos leer otro tipo deberemos convertirlo. Por ejemplo:

```py
a = int (input("Introduce el valor de a"))
a = float (input("Introduce el valor de a"))
a = bool (input("Introduce el valor de a"))
```

Si queremos convertir una variable en una cadena haremos lo siguiente:

```py
a = str(a)
```

## 1.6 Variables alfanuméricas.
### 1.6.1 Operaciones con cadenas.
#### 1.6.1.1 Suma o concatenación de cadenas.
Utilizaremos el símbolo `+` entre las cadenas. Por ejemplo:

```py
>>> a = "Azul"; b = "Blanco"
>>> a + b
'AzulBlanco'

# Otra forma

>>> 'Azul' 'Blanco'
'AzulBlanco'
```

#### 1.6.1.2 Multiplicación de cadenas.
Se realiza entre una cadena y una variable.

```py
>>> a = 'Hola'

# Si la variable es entera
>>> 3*a
'HolaHolaHola'  # repetición de la cadena por el valor del entero

# Si la variable es lógica
>>> a*True
'Hola'

>>> a*False
''

>>> a*(not True)
''

>>> a*0
''

>>> a*''
''
```

## 1.7 Listas.
Se definen entre corchetes. Por ejemplo:

```py
>>> colores = ['Blanco', 'Azul', 'Rojo']
```

Podemos realizar las mismas operaciones que con las cadenas.

Para calcular la longitud, utilizaremos `len(colores)`.

### 1.7.1 Diccionarios.
Son listas formadas por pares. Cada par está formada por una llave (key) o clave y su valor. La llave y su valor se separan por dos puntos. Los pares se separan por comas. Los pares van encerrados entre llaves. Por ejemplo:

```py
>>> diccionario = {'a' : 23, 'luna' : 'unica', 'x' : 7.3}
```

Para imprimir los valores de un par usamos:

```py
>>> print(diccionario['a'])
23
```

Un diccionario también puede contener como elementos listas. Por ejemplo:

```py
x = {"Álgebra" : [20, 30, 40], "Aritmética" : [17, 41], "Cálculo" : [9]}
```

Para desplegar un elemento usamos:

```py
>>> x["Álgebra"]
[20, 30, 40]
```
