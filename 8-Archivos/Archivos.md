# 8. Archivos
## 8.1 Escritura de datos en un archivo
El formato es el siguiente: `identificador = open(nombre_del_archivo, modo)`, donde `identificador` es una variable para referenciar al archivo y la variable `modo` puede tener alguna de las siguientes opciones:
- `r`: abre el archivo para lectura.
- `w`: abre el archivo para escribir. Si ya existe, borra los datos existentes.
- `a`: abre el archivo para escribir. Si ya existe, escribe después de los datos existentes.
- `r+`: abre el archivo para leer y escribir.

### 8.1.1 Escritura de datos alfanuméricos
Una vez abierto el archivo, el siguiente paso es escribir en él con la instrucción `write` (solo podemos escribir cadenas y NO genera salto de línea): `identificador.write('frase')`.

También podemos utilizar la instrucción `writelines`: `identificador.writelines(["frase1", ..., "fraseN"])`.

Por último, tenemos que cerrar el archivo: `identificador.close()`.

### 8.1.2 Instrucción `with`
Proporciona otra manera de abrir un archivo para escritura o lectura y el archivo se cierra después de usarlo. El formato es:

```py
with open("nombre_del_archivo.txt", "modo") ad identificador:
    identificador.write("frase")
```

## 8.2 Escritura de datos numéricos
Tenemos que convertir los datos numéricos a cadenas usando `str()` y aplicar el apartado anterior.

Si queremos crear una **tabla** en un archivo, en Python esa "tabla" tiene que ser una matriz. Por ejemplo:

```py
datos = [   [1.2, 3.0, -7.45],
            [-8.3, 6, 17.563],  
            [98.78, -12.5, -46.2332],
            [2, -567.2, 8.43],
            [1, 0, 1.2013] ]

tabla_de_datos = open("tabla.dat", "w")

for renglon in datos:
    for columna in renglon:
        tabla_de_datos.write("%14.8f", %columna)

    tabla_de_datos.write("\n")

tabla_de_datos.close()
```

## 8.3 Lectura de datos de un archivo
Para leer datos de un archivo necesitamos saber el formato de los datos. Primero tenemos que abrir el archivo con: `identificador = open("nombre_del_archivo", "r")`. Veamos los distintos casos que se pueden presentar.

- Si el archivo solo tiene un dato en cada línea: podemos leer dichos datos con `for linea in identificador` y en `linea` se almacenarán los datos de cada línea.
- Si el archivo tiene más de un dato en cada línea: podemos leer dichos datos con `for linea in identificador` y luego separar la línea con `datos = linea.split()`, por lo que si accedemos ahora así: `datos[i]`, podemos leer cada dato de la línea.

Hay que cerrar el archivo después de leerlo.

## 8.4 Lectura y escritura de datos en Excel
Uno de los formatos para guardar Excel es `csv` y para que Python pueda leer los valores, hay que importar la biblioteca `csv` que contiene las funciones `csv.reader` y `csv.writer`.

### 8.4.1 Lectura de datos en Excel
Hay que seguir los siguientes pasos:

1. Importar la biblioteca `csv`: `import csv`.
2. Abrir el archivo con: `archivo = open("excel.csv", "r")`.
3. Inicializar la lista donde vamos a guardar cada línea y el resultado será una lista de listas: `lista []`.
4. Usamos el ciclo `for` para leer las líneas. Cada línea se añade a la lista con `append`:

```py
for linea in csv.reader(archivo):
    lista.append(linea)
```

5. Al terminar de leer el archivo, lo cerramos con: `archivo.close()`.
6. Para ver los datos leídos, ya que se trata de una lista de listas, usaremos la instrucción `pprint`:

```py
import pprint
pprint.pprint(tabla)
```

### 8.4.2 Escritura de datos en Excel
Usaremos la instrucción `csv.writer`. Seguiremos los siguientes pasos:

Supongamos que tenemos una lista de listas con el nombre de `lista`.

1. Importar la biblioteca `csv`: `import csv`.
2. Crear el archivo para escribir y definir un identificador: `identificador = open("excel.csv", "w")`.
3. Crear un identificador para escribir en nuestro archivo con la instrucción `writer`: `identificador_escribir = csv.writer(identificador)`.
4. Escribir las líneas con un ciclo `for` y con la instrucción `writerow`:

```py
for linea in lista:
    identificador_escribir.writerow(linea) # writerow escribe una línea completa en un archivo
```

5. Cerrar el archivo: `identificador.close()`.
