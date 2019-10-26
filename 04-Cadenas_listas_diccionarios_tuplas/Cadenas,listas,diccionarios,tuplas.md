# 4. Cadenas, listas, diccionarios y tuplas
## 4.1 Cadenas
Las cadenas son elementos alfanuméricos que se encierran entre comillas dobles o simples. El primer carácter de una cadena de tamaño `n` ocupa la posición 0, y el último ocupa la posición `n-1` o -1.

Para acceder a un elemento en la posición `k` haremos `cadena[k]`.

Si queremos solo una porción de una cadena, es decir, seleccionar los elementos del k-ésimo al n-ésimo haremos `cadena[k : n+1]`.

La porción desde el índice `k` hasta el final de la palabra se hace con `cadena [k : ]`.

La porción desde el inicio de la cadena hasta el índice `k` se hace con `cadena [ : k+1]`.

### 4.1.1 Longitud de una cadena
Utilizaremos la instrucción `len(cadena)`.

### 4.1.2 Separación de una cadena
Utilizaremos la instrucción `split`. Por ejemplo: `cadena.split(subcadena)`, donde `subcadena` es una porción de la cadena. Las cadenas resultantes se almacenan en una lista. Si la `subcadena` es un espacio en blanco, entonces la cadena se separa en los espacios entre las palabras de la cadena. Pero si escribimos una subcadena, la cadena se separa en porciones separadas por la subcadena. Por ejemplo:

Consideramos la cadena `Sagan`:

`Sagan = "Vivimos en una sociedad profundamente dependiente de la ciencia y la tecnología."`

Si usamos:

```py
>>> Sagan.split( )
['Vivimos','en','una','sociedad','profundamente','dependiente','de','la','ciencia','y','la','tecnología']
```

Si separamos con la sílaba `te` tenemos:

```py
>>> Sagan.split('te')
['Vivimos en una sociedad profundamen', ' dependien', ' de la ciencia y la ', 'cnología.']
```

### 4.1.3 Inmutabilidad de las cadenas
Una vez se crea una cadena, ninguno de sus caracteres puede cambiar.

### 4.1.4 Otras operaciones con cadenas
La forma de usar las siguientes funciones es `cadena.funcion()`.

- `capitalize()`: convierte el primer caracter a mayúscula.
- `center(N, 'x')`: pone `x` a ambos lados de la cadena hasta acompletar `N` caracteres. La cadena original queda en el centro.
- `count(x)`: cuenta cuantas veces aparece el caracter `x`.
- `upper()`: convierte minúsculas a mayúsculas.
- `swapcase()`: invierte mayúsculas a minúsculas y viceversa.
- `endswith(suf)`: devuelve `true` si la cadena termina con el sufijo `suf`.
- `expandtabs(tab = n)`: cambia tabuladores por `n` espacios.
- `find(subcadena)`: determina si la subcadena es parte de la cadena.
- `title()`: cambia a mayúsculas el primer elemento.
- `isalnum()`: devuelve `true` si los caracteres son alfanuméricos.
- `isalpha()`: devuelve `true` si todos los caracteres son alfanuméricos.
- `isdigit()`: devuelve `true` si todos los caracteres son dígitos.
- `islower()`: devuelve `true` si tiene solo letras minúsculas.
- `isnumeric()`: devuelve `true` si solo tiene números.
- `isspace()`: devuelve `true` si solo tiene espacios.
- `istitle()`: devuelve `true` si empieza con mayúscula.
- `isupper()`: devuelve `true` si solamente tiene mayúsculas.
- `lower()`: convierte mayúsculas a minúsculas.
- `max(cadena)`: obtiene el caracter de mayor valor.
- `min(cadena)`: obtiene el caracter de menor valor.
- `replace ("a", "b")`: reemplaza el caracter "`a`" con "`b`".
- `rstrip()`: remueve los espacios de la derecha.
- `lstrip()`: remueve los espacios de la izquierda.
- `strip()`: realiza `lstrip()` y `rstrip()`.
- `zfill(N)`: añade ceros al inicio para tener `N` caracteres.
- `join`: une cadenas para formar una lista de cadenas.

## 4.2 Listas
Son colecciones de datos los cuales pueden ser de cualquier tipo. La estructura de las listas es: `lista [a, b, c, ...]`. Se pueden concatenar y sumar, como las cadenas. Además, no son inmutables.

Si queremos convertir una lista de cadenas en una cadena haremos `subcadena.join([lista de cadenas])`, donde la `subcadena` especifica con qué se van a separar las cadenas. Por ejemplo:

```py
lista = ["perros", "gatos", "ratones"]
subcadena = [" y "] # Notar que hay un espacio antes y después de la y

>>> subcadena.join(lista) # Unimos los elementos de la lista
'perros y gatos y ratones'
```

### 4.2.1 Otras operaciones con listas
- `tuple(lista)`: convierte una lista a tupla.
- `lista.append(obj)`: adiciona objetos a la lista.
- `lista.count(obj)`: devuelve cuántas veces `obj` está en la lista.
- `lista1.extend(lista2)`: adiciona `lista2` al final de `lista1`.
- `lista.index(objeto)`: devuelve el índice del primer elemento que es igual a `objeto`.
- `lista.insert(índice, obj)`: inserta `obj` en la posición `índice`.
- `lista.pop(índice)`: devuelve y elimina de la lista el elemento. Si no se da un índice, devuelve el último elemento.
- `lista.remove(objeto)`: elimina `objeto` de la lista. Si no existe aparece un error.
- `lista.reverse()`: invierte el orden de los objetos de la lista.
- `lista.sort()`: ordena los elementos de la lista.

## 4.3 Tuplas
Es una estructura de datos conformada por elementos los cuales pueden ser de distinto tipo. Son inmutables. La estructura de las tuplas es: `tupla(a, b, c, ...)`

Se pueden **intercambiar los valores** de las variables más fácil haciendo
```py
>>> (a, b) = (b, a)
```

Otras operaciones con las tuplas son:
- `tuple(lista)`: convierte una lista a tupla.
- `len(tupla)`: longitud de una tupla.
- `max(tupla)`: devuelve el valor máximo de la tupla.
- `min(tupla)`: devuelve el valor mínimo de la tupla.

## 4.4 Diccionarios
Son estructuras de datos consistentes en **listas de pares** de variables. Cada par tiene un elemento llamado **clave (key)** que puede ser de cualquier tipo y otro elemento llamado **valor** que también puede ser de cualquier tipo. El formato de un diccionario es:

`diccionario = { "clave1":valor1, "clave2":valor2, ..., "claveM":valorM}`

No son inmutables. Además, puede darse el caso de que Python cambie el orden de dichos pares.

Otras operaciones con los diccionarios son:
- `str(diccionario)`: convierte el diccionario en cadena.
- `type(variable)`: devuelve el tipo de la variable.
- `dicc.clear()`: elimina todos los elementos.
- `dicc.copy()`: copia el diccionario en otra variable.
- `dicc.fromkeys(lista, valor)`: crea un nuevo diccionario. Las claves son de lista con valor.
- `dicc.get(clave)`: devuelve el valor de la clave.
- `dicc.has_key(clave)`: devuelve `true` si la clave está en el diccionario.
- `dicc.items()`: devuelve una lista de pares (clave, valor).
- `dicc1.update(dicc2)`: añade los pares clave-valor de dicc2 al dicc1.
- `dicc.values()`: devuelve una lista con los valores.
