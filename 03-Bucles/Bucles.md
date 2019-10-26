# 3. Bucles
## 3.1 Bucles While
El formato de la instrucción `while` es la siguiente.

```py
while (expresión lógica):
    instrucciones
```

## 3.2 Bucles For
La estructura del ciclo `for` es la siguiente.

```py
for contador in lista:
    instrucciones
```

NO deben usarse paréntesis en la línea del `for`. Tampoco es necesario inicializar el `contador`, ya que se hace dentro de la primera instrucción dentro de la condición. `contador` se inicializa con el primer valor de la `lista`. El último valor de la `lista` indica hasta qué valor del contador se realizará el ciclo. El contador se incrementa en la unidad de acuerdo al valor en la `lista`.

### 3.2.1 Función `range`
Esta función también se puede usar en lugar de una lista. Tiene el formato `range (a, b, c)`.

Esta función crea una lista que empieza en `a`, termina antes de `b`, `c` es el incremento y los valores de la lista generada son: `[a, a + c, a +2c, a + 3c,..., ]`. Hay que tener en cuenta que el valor `b` NO forma parte de la lista. Además, el valor `c` NO puede ser 0.

Por ejemplo:

`range (5)` produce `[0, 1, 2, 3, 4]`

`range (2, 6)` produce `[2, 3, 4, 5]`

`range (-1, 7, 2)` produce `[-1, 1, 3, 5]`

Para desplegar los valores de una lista generada con `range` usamos la instrucción `list`, por ejemplo: `list (range(4))`.

## 3.3 Bucles Anidados
Los bucles `while` y `for` se pueden anidar.

## 3.4 Instrucción `continue`
Se utiliza dentro de un bucle `for` asociado a un bucle `if` y no ejecuta el renglón de código que está después de que aparece la instrucción `continue`. Por lo general, se usa después de una condición `if`.

## 3.5 Instrucción `break`
Se utiliza en los bucles `for` y los interrumpe (los termina).

## 3.6 Instrucciones para números aleatorios
- `randint`: genera números aleatorios enteros.
- `random` : bibioteca con funciones que generan números aleatorios.
