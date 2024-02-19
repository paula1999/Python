# DataFrame Fundamentos

## Empezar una sesión

```py
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("Sesion").getOrCreate()
```

## Leer un archivo

```py
df = spark.read.json('archivo.json')
```

## Leer del storage account

```py
path = f'abfss://default@stabd{env}neu{name}.dfs.core.windows.net'
table = '/app/table'
spark.read.format("delta").load(
    path + table
)
```

## Leer del catálogo

```py
path = f'{env}_{name}.{esquema_nombre}.table'
spark.read.format("delta").load(
    path
)
```

## Mostrar los datos

```py
df.show() # Datos en forma de tabla
df.printSchema() # Esquema
df.columns # Columnas
df.describe() # Tipos de dato de las columnas
```

## Crear un DataFrame especificando el esquema

```py
from pyspark.sql.types import StructField,StringType,IntegerType,StructType

data_schema = [StructField("col1", IntegerType(), True), # nombre de la columna, tipo de dato y si puede se nulo
                StructField("col2", StringType(), True)]
final_struc = StructType(fields = data_schema)
df = spark.read.json('archivo.json', schema = final_struc)
```

## Acceso a los datos

```py
df.select('col').show() # datos de una columna
df.head(2) # lista con las dos primeras filas
df.select(['col1', 'col2']).show() # datos de dos columnas
```

## Crear columnas

```py
df.withColumn('nueva_columna', df['columna'] * 2).show() # añadir columna a partir de una operación con otra columna
df.withColumnRenamed('columna', 'nueva_columna').show() # renombrar columna
```

## SQL

Para usar consultas de SQL directamente en un dataframe, necesitamos registrarlo como una vista temporal:

```py
df.createOrReplaceTempView("archivo")
```

```py
spark.sql("SELECT * FROM archivo").show()
```


## Guardar dataframe

### Storage

```py
(df_table.write
    .mode("overwrite")
    .format("delta")
    # .option("overwriteSchema", "true")
    .save(save_path + table)
)
```

### Catálogo

```py
df_save_list = [
    {
      "df": df_table,
      "table_name": "table"
    }
]

app_path = f'{env}_{name}.{esquema_nombre}.table'

for idx, aux_dict in enumerate(df_save_list):
    print(f"Guardando {aux_dict['table_name']} - {idx + 1}/{len(df_save_list)}")
    t_0 = time.time()
    file_path = app_path + aux_dict["table_name"]
    # spark.sql(f"DROP TABLE IF EXISTS {env}_{name}.{esquema_nombre}.{aux_dict['table_name']}")
    (aux_dict["df"].write.format("delta")
        .mode("overwrite")
        # .option("overwriteSchema", True)
        .option('path', file_path)
        .saveAsTable(f'{env}_{name}.{esquema_nombre}.{aux_dict["table_name"]}')
        )
    print(f"Tiempo en guardar en catálogo Gold: {(time.time() - t_0):.3f} segundos")
```
