# DataFrame Valores perdidos

## Eliminar valores perdidos

Se usa el siguiente comando:

```py
df.na.drop(how = 'any', thresh = None, subset = None)
```

donde:

- `how`
  - `any`: eliminar la fila si contiene algún nulo.
  - `all`: eliminar la fila si todos los valores son nulos.
- `thresh`
  - Toma valores `int`, pero por defecto es `None`.
  - Elimina las filas que tienen menos `thresh` valores no nulos.
- `subset`
  - Lista de las columnas a considerar.

## Rellenar los valores perdidos

```py
df.na.fill('nuevo valor').show() # rellenar todas las columnas
df.na.fill('nuevo valor', subset = ['col']).show() # rellenar solo la columna col
```

Se suele rellenar los valores perdidos con el valor de la media:

```py
from pyspark.sql.functions import mean

mean_val = df.select(mean(df['col'])).collect()
mean_sales = mean_val[0][0]
df.na.fill(mean_sales,["col"]).show()
```
