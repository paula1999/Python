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
