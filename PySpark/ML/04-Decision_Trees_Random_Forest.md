# Decision Trees y Random Forest

```py
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer

# Load training data
data = spark.read.csv('archivo.csv', inferSchema = True, header = True)

# Transform
assembler = VectorAssembler(inputCols = ["col1", "col2"], outputCol = "features")
df = assembler.transform(data)

# String to number
indexer = StringIndexer(inputCol = "col", outputCol = "new_col")
df_fixed = indexer.fit(df).transform(df)

train_data,test_data = final_data.randomSplit([0.7,0.3])
```


```py
from pyspark.ml.classification import DecisionTreeClassifier, GBTClassifier, RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Models
dtc = DecisionTreeClassifier(labelCol = 'new_col', featuresCol = 'features')
rfc = RandomForestClassifier(labelCol = 'new_col', featuresCol = 'features')
gbt = GBTClassifier(labelCol = 'new_col', featuresCol = 'features')

# Train the models
dtc_model = dtc.fit(train_data)
rfc_model = rfc.fit(train_data)
gbt_model = gbt.fit(train_data)

# Model comparison
dtc_predictions = dtc_model.transform(test_data)
rfc_predictions = rfc_model.transform(test_data)
gbt_predictions = gbt_model.transform(test_data)

# Select (prediction, true label) and compute test error
acc_evaluator = MulticlassClassificationEvaluator(labelCol = "new_col", predictionCol = "prediction", metricName = "accuracy")

dtc_acc = acc_evaluator.evaluate(dtc_predictions)
rfc_acc = acc_evaluator.evaluate(rfc_predictions)
gbt_acc = acc_evaluator.evaluate(gbt_predictions)
```
