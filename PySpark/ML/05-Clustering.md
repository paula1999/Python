# Clustering K-Means

## Input Columns

| Parámetro | Tipo | Por defecto | Descripción |
|---|---|---|---|
| featuresCol | Vector | "features" | Vector de las características |


## Output Columns

| Parámetro | Tipo | Por defecto | Descripción |
|---|---|---|---|
| predictionCol | Int | "prediction" | Centro del cluster predicho |

```py
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import StandardScaler

dataset = spark.read.format("libsvm").load("archivo.txt")

# Format the data
vec_assembler = VectorAssembler(inputCols = dataset.columns, outputCol = 'features')
final_data = vec_assembler.transform(dataset)

# Scale the data
scaler = StandardScaler(inputCol = "features", outputCol = "scaledFeatures", withStd = True, withMean = False)

# Compute summary statistics by fitting the StandardScaler
scalerModel = scaler.fit(final_data)

# Normalize each feature to have unit standard deviation.
final_data = scalerModel.transform(final_data)

# Train the model and evaluate

# Trains a k-means model
kmeans = KMeans(featuresCol = 'scaledFeatures', k = 3)
model = kmeans.fit(final_data)

# Evaluate clustering by computing Within Set Sum of Squared Errors
wssse = model.computeCost(final_data)
print("Within Set Sum of Squared Errors = " + str(wssse))

# Shows the result
centers = model.clusterCenters()
print("Cluster Centers: ")
for center in centers:
    print(center)
    
model.transform(final_data).select('prediction').show()
```
