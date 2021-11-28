# Customer Churn Prediction with PySpark

from pyspark.ml.classification import GBTClassifier, LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder, StandardScaler
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql import SparkSession
from pyspark.ml.feature import Bucketizer
from pyspark.sql.functions import when, count, col
import warnings
import findspark
import pandas as pd

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

findspark.init(r"...\spark")  # spark file location here

# we're creating a session
spark = SparkSession.builder \
    .master("local") \
    .appName("pyspark_giris") \
    .getOrCreate()

sc = spark.sparkContext

##################################################
# Exploratory Data Analysis
##################################################
spark_df = spark.read.csv(r"...\churn.csv",  # file location here
                          header=True, inferSchema=True)

print("Shape: ", (spark_df.count(), len(spark_df.columns)))
# 900,7

# feature types
spark_df.printSchema()

spark_df.show()

# lower case
spark_df = spark_df.toDF(*[c.lower() for c in spark_df.columns])

# summary
spark_df.describe().show()

# numerical feature selection and summaries
num_cols = [col[0] for col in spark_df.dtypes if col[1] != 'string']
spark_df.select(num_cols).describe().show()

# catogorical feature selection and summaries
cat_cols = [col[0] for col in spark_df.dtypes if col[1] == 'string']
for col in cat_cols:
    spark_df.select(col).distinct().show()

for col in num_cols:
    spark_df.groupby("churn").agg({col: "mean"}).show()

##################################################
# SQL Queries
##################################################

spark_df.createOrReplaceTempView("tbl_df")

spark.sql("show databases").show()

spark.sql("show tables").show()

spark.sql("select age from tbl_df limit 5").show()

spark.sql("select churn, avg(age) from tbl_df group by Churn").show()

##################################################
# Data Preprocessing & Feature Engineering
##################################################

# Missing Values

# we're dropping missing values
spark_df = spark_df.dropna()

# Feature Interaction

# deriving new features
spark_df = spark_df.withColumn('age_total_purchase', spark_df.age / spark_df.total_purchase)
spark_df.show(5)


# Bucketization / Bining
############################

bucketizer = Bucketizer(splits=[0, 35, 45, 65], inputCol="age", outputCol="age_cat")

spark_df = bucketizer.setHandleInvalid("keep").transform(spark_df)

spark_df.show(20)

spark_df = spark_df.withColumn('age_cat', spark_df.age_cat + 1)


# Derived Features
############################

spark_df = spark_df.withColumn('segment', when(spark_df['years'] < 5, "segment_b").otherwise("segment_a"))

spark_df.withColumn('age_cat_2',
                    when(spark_df['age'] < 36, "young").
                    when((35 < spark_df['age']) & (spark_df['age'] < 46), "mature").
                    otherwise("senior"))

# Label Encoding
############################

spark_df.show(5)

indexer = StringIndexer(inputCol="segment", outputCol="segment_label")
temp_sdf = indexer.fit(spark_df).transform(spark_df)
spark_df = temp_sdf.withColumn("segment_label", temp_sdf["segment_label"].cast("integer"))
spark_df = spark_df.drop('segment')


# One Hot Encoding
############################

encoder = OneHotEncoder(inputCols=["age_cat"], outputCols=["age_cat_ohe"])
spark_df = encoder.fit(spark_df).transform(spark_df)

spark_df.show(5)

# Target & Feature Definition
############################

stringIndexer = StringIndexer(inputCol='churn', outputCol='label')
temp_sdf = stringIndexer.fit(spark_df).transform(spark_df)
spark_df = temp_sdf.withColumn("label", temp_sdf["label"].cast("integer"))


cols = ['age', 'total_purchase', 'account_manager', 'years',
        'num_sites', 'age_total_purchase', 'segment_label', 'age_cat_ohe']

va = VectorAssembler(inputCols=cols, outputCol="features")
va_df = va.transform(spark_df)
va_df.show()

# Final df
final_df = va_df.select("features", "label")
final_df.show(5)

# Split the dataset into test and train sets.
train_df, test_df = final_df.randomSplit([0.7, 0.3], seed=17)
train_df.show(10)
test_df.show(10)

print("Training Dataset Count: " + str(train_df.count()))
print("Test Dataset Count: " + str(test_df.count()))

##################################################
# Modeling
##################################################
############################
# Gradient Boosted Tree Classifier
############################

gbm = GBTClassifier(maxIter=100, featuresCol="features", labelCol="label")
gbm_model = gbm.fit(train_df)
y_pred = gbm_model.transform(test_df)
y_pred.show(5)

y_pred.filter(y_pred.label == y_pred.prediction).count() / y_pred.count()
# 0.8469750889679716

############################
# Model Tuning
############################

evaluator = BinaryClassificationEvaluator()

gbm_params = (ParamGridBuilder()
              .addGrid(gbm.maxDepth, [2, 4, 6])
              .addGrid(gbm.maxBins, [20, 30])
              .addGrid(gbm.maxIter, [10, 20])
              .build())

cv = CrossValidator(estimator=gbm,
                    estimatorParamMaps=gbm_params,
                    evaluator=evaluator,
                    numFolds=5)

cv_model = cv.fit(train_df)


y_pred = cv_model.transform(test_df)
ac = y_pred.select("label", "prediction")
ac.filter(ac.label == ac.prediction).count() / ac.count()
# 0.8896797153024911

sc.stop()
