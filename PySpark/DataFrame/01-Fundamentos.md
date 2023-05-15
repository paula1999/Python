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
