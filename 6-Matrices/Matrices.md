# 6. Matrices
## 6.1 Matrices
Se escriben como listas de listas: `matriz[ [a, b, c, ...], [a', b', c', ...], ..., ]`. Además, si queremos generar una matriz vacía y posteriormente llenarlo con los datos requeridos podemos hacer lo siguiente:

```py
A = []

for i in range(m):
    A.append([]) # Se genera un renglón

    for j in range(n):
        A[i].append(None) # Genera cada uno de los elementos
```

También podemos definirla mediante el **método de comprensión**: ` matriz = [ [f(i) for i in range(columnas)] for j in range(renglones) ]`.

## 6.2 Métodos alternos de escritura de matrices
### 6.2.1 Uso de iteradores de listas
Usaríamos:

```py
for renglon in matriz:
    print(renglon)
```

Otra forma sería:

```py
for renglon in matriz:
    print(renglon)

    for c in renglon:
        print(c, end = ', ') # Para separar las columnas
```


## 6.3 Selección de filas y columnas
Se accede a las filas y/o columnas con los corchetes `[]` (`matriz[i][j]`).
