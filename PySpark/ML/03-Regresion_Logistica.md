# Regresión Logística

```py
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler

# Load training data
data = spark.read.format("libsvm").load("archivo.txt")

# Transform
assembler = VectorAssembler(inputCols = ['col1', 'col2'], outputCol = 'features')
df = assembler.transform(data)

training, test = df.randomSplit([0.7,0.3])

lr = LogisticRegression(labelCol = "label")

# Fit the model
lrModel = lr.fit(training)

trainingSummary = lrModel.summary

trainingSummary.predictions.describe().show()
```

```py
from pyspark.mllib.evaluation import MulticlassMetrics

predictionAndLabels = lrModel.evaluate(test)
predictionAndLabels.predictions.show()
```

## Evaluadores

```py
from pyspark.ml.evaluation import BinaryClassificationEvaluator,MulticlassClassificationEvaluator

evaluator = BinaryClassificationEvaluator(rawPredictionCol = 'prediction', labelCol = 'label')
# For multiclass
evaluator = MulticlassClassificationEvaluator(predictionCol = 'prediction', labelCol = 'label', metricName = 'accuracy')

acc = evaluator.evaluate(predictionAndLabels)
auc = evaluator.evaluate(predictionAndLabels.predictions)
```
