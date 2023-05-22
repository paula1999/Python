# Recomendaciones

```py
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS

data = spark.read.csv('archivo.csv', inferSchema = True, header = True)

(training, test) = data.randomSplit([0.8, 0.2])

# Build the recommendation model using ALS on the training data
als = ALS(maxIter = 5, regParam = 0.01, userCol = "userId", itemCol = "itemId", ratingCol = "rating")
model = als.fit(training)

# Evaluate the model by computing the RMSE on the test data
predictions = model.transform(test)

evaluator = RegressionEvaluator(metricName = "rmse", labelCol = "rating", predictionCol = "prediction")
rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = " + str(rmse))
```

```py
# Recommendation to a user
single_user = test.filter(test['userId']==11).select(['itemId', 'userId'])
reccomendations = model.transform(single_user)
reccomendations.orderBy('prediction', ascending = False).show()
```
