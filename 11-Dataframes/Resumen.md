# Leer

```py
df = pd.read_csv("archivo.csv", header = None)
```

# Seleccionar columnas

```py
df[["col1", "col2"]]
```


# Renombrar columnas

```py
df = df.rename(
  columns = {
    "col1" : "new_col1"
  }
)
```

# Descripción df

```py
df.info()
df.corr()
```

# Filtrar

```py
df[df["col"] == "x"]
```

# Tamaño

```py
df.shape[0]
```

# Mínimo

```py
df["col"].min()
```


# Crear df

```py
df = pd.DataFrame(columns = {"col1", "col2"})
df.loc[0] = ["A", "B"]
```

# Fechas

```py
pd.to_datetime(df["col"], format = "%Y-%m-%d %H:%M:%S")

df["col"].dt.hour

df["col"].dt.day
```

# Índice

```py
df.set_index("col", inplace = True)
```

# Ordenar valores

```py
df.sort_values("col", ascending = False)
```

# Agrupar

```py
df["col"].value_counts().to_frame()
```

```py
df.groupby(["col1"])["col2"].count().to_frame()
```

# Top 10

```py
df[:10
```

# Merge

```py
df1.merge(
  df2,
  how = "left",
  left_on = "left_col",
  right_on = "right_col"
)
```


# Gráfica

```py
df.plot(kind = "bar")
```







