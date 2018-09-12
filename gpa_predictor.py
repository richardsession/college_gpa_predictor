from pyspark.sql import SQLContext
from pyspark import SparkContext
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vectors
from pyspark.sql.types import *
from flask import Flask, request

sc = SparkContext()
sqlContext = SQLContext(sc)

app = Flask(__name__)

# Converts file to DataFrame
gpa_df = sqlContext.read.load("./gpa_data.csv",
    format='com.databricks.spark.csv',
    header='true',
    inferSchema='true')

# Begin setting up linear regression
lr = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)

# Features must be vectorized
lr.setFeaturesCol("hs_gpa_vector")
lr.setLabelCol("c_gpa")

# Vectorizes a column within the DataFrame
assembler = VectorAssembler(inputCols=["hs_gpa"],outputCol="hs_gpa_vector")

# Pass in the DataFrame to transform the specified column into vectors. Added to new dataset with 3rd column called hs_gpa_vector
output = assembler.transform(gpa_df)

# 70%/30% split
split = output.randomSplit([0.7, 0.3])

model = ''

@app.route('/home')
def doHome():
	return 'Hello, World!'

# Train model with the 70% of data
@app.route('/train')
def doTrain():
	training = split[0]

	# With a linear regression, the fit method trains the model using the specified data set
	model = lr.fit(training)
	return "It's Training!"

@app.route('/predict')
def doPredict():
	# Manually create a DataFrame without loading a document. Must use vectorized representation of input field (in this case hs_gpa)
	hs_gpa = sqlContext.createDataFrame([(Vectors.dense(float(request.args.get('gpa'))))], ['hs_gpa_vector'])

	# Use the model and pass in DataFrame in order to predict the college gpa (c_gpa). transform = predict
	predictions = model.transform(hs_gpa)
	
	return predictions.show()

app.run(host='0.0.0.0', port=5002, debug=True, threaded=True)