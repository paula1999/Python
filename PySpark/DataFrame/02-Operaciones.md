# DataFrame Operaciones

## Filtrar

```py
# Filtrar los datos de una columna numérica que sean menores que 500
df.filter("col < 500").show()
df.filter(df["col"] < 500).show() # usando operadores de comparación python
```


```py
# varios operadores
df.filter( (df["col"] > 200) & (df["col"] < 500) ).show()
```

```py
# Recolectar los resultados como objetos de python
result = df.filter(df["col"] == 200).collect()

# Las filas se pueden convertir en diccionarios
row = result[0]
row.asDict()
```

## Agrupar

```py
df.groupBy("col").mean().show() # media de cada grupo
df.groupBy("col").count().show() # número de elementos de cada grupo
df.groupBy("col").max().show() # máximo de cada grupo
df.groupBy("col").min().show() # mínimo de cada grupo
df.groupBy("col").sum().show() # suma total de cada grupo
```

No todos los métodos necesitan llamar a `groupBy()`, sino que también se puede usar el método `agg()`:

```py
df.agg({'col' : 'max'}).show() # máximo absoluto de la columna
```

## Funciones

Se importan de `pyspark.sql.functions`.

```py
from pyspark.sql.functions import countDistinct, avg, stddev
```

```py
df.select(countDistinct("col")).show()

# Para cambiar el nombre de la columna
df.select(countDistinct("col").alias("nueva_columna")).show()
```

```py
df.select(avg('col')).show()
df.select(stddev("col")).show()
```

## Ordenar

```py
df.orderBy("col").show() # ascendente
df.orderBy(df["col"].desc()).show() # descendente
```
