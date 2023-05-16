# Regresión Lineal

```py
from pyspark.ml.regression import LinearRegression

# Load data
data = spark.read.format("libsvm").load("archivo.txt")
data = spark.read.csv("archivo.csv", inferSchema = True, header = True)
```

Spark espera que el formato de los datos sean dos columnas: `label` y `features`.

- La columna `label` toma valores numéricos.
- La columna `features` tiene un vector de todas las características que pertenecen a esa fila.

```py
train_data, test_data = all_data.randomSplit([0.7,0.3])

lr = LinearRegression(featuresCol = 'features', labelCol = 'label', predictionCol = 'prediction') # default values

# Fit the model
lrModel = lr.fit(train_data)

# Print the coefficients and intercept for linear regression
print("Coefficients: {} Intercept: {}".format(lrModel.coefficients, lrModel.intercept))

# Evaluate
test_results = correct_model.evaluate(test_data)

# Summarize the model over the training set and print out some metrics
test_results.residuals.show()
print("RMSE: {}".format(test_results.rootMeanSquaredError))
print("MSE: {}".format(test_results.meanSquaredError))
print("r2: {}".format(test_results.r2))

# Predictions
unlabeled_data = test_data.select('features')
predictions = correct_model.transform(unlabeled_data)
```
