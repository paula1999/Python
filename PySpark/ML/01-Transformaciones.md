# Transformaciones de los datos

## Características de los datos

### StringIndexer

Se usa para convertir información de tipo string en información numérica como una catacterística categórica.

```py
from pyspark.ml.feature import StringIndexer

indexer = StringIndexer(inputCol = "col_category", outputCol = "col_num")
new_df = indexer.fit(df).transform(df)
```

### VectorIndexer

`VectorAssembler` transforma una lista de columnas en una única columna de vectores.

| col1 | col2 | col3 |
|---|---|---|
| a | b | c |


```py
from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(inputCols = ["col1", "col2", "col3"], outputCol = "col")
new_df = assembler.transform(df)
```

| col |
|---|
| [a, b, c]|

### Transformación de variables numéricas y categóricas

```py
def transform_data(data, categorical_cols, numerical_col, def_model):
    # Transformación de variables categóricas
    indexers = [ft.StringIndexer(inputCol=col, outputCol=col + "_index", handleInvalid="keep") for col in categorical_col]
    encoders = [ft.OneHotEncoder(inputCol=col + "_index", outputCol=col + "_encoded") for col in categorical_col]

    # Ensamblar todas las características en un solo vector
    assembler = ft.VectorAssembler(inputCols=[col + "_encoded" for col in categorical_col] + numerical_col, outputCol="features")

    # Crear el pipeline
    pipeline = ml.Pipeline(stages=indexers + encoders + [assembler, def_model])

    return pipeline
```
