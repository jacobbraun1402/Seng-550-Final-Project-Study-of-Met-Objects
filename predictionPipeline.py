from pyspark.sql import SparkSession
import os
import subprocess
from pyspark.sql.functions import col, udf, when
from pyspark.sql.types import IntegerType
from pyspark.ml.feature import VectorAssembler, Tokenizer, Word2Vec, OneHotEncoder, StringIndexer
from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel

# function that applies all the transformations needed to the dataset to perform model training
def transformDataset(df):
    # in the database, they only store a value in the gender field if the artist is female
    # otherwise it is null
    # if there are multiple artists attributed to an object, the gender of each artist is shown, separated by a '|'
    df = genderColumn(df)

    # drop culture because it has a ton of nulls, I think artist nationality along with the department it's from and year
    # it was made will still allow you to somewhat infer the culture it came from
    df = df.drop('culture')
    # can't have null values in model
    df = df.dropna()

    # change nationality column into word2Vec encoding
    df = vectorizeNationality(df)

    # change object name column into word2vec encoding
    df = vectorizeObject(df)

    #change department to one hot encoding, since this is categorical
    df = oneHotDepartment(df)

    #change isHighlight and isTimelineWork to integers
    df = df.withColumn('isHighlightInt', when(df.isHighlight == True, 1).otherwise(0)).drop('isHighlight')
    df = df.withColumn('isTimelineWorkInt', when(df.isTimelineWork == True, 1).otherwise(0)).drop('isTimelineWork')

    # cast accession year and object end date data types to integers

    df = df.withColumn('accessionYearInt', df.accessionYear.cast(IntegerType())).drop('accessionYear')\
         .withColumn('objectDate', df.objectEndDate.cast(IntegerType())).drop('objectEndDate')\
          .drop('objectID')

    
    return df

#create custom function to convert gender column to boolean indicating if artist is female
# 0=Male, 1=Female
# in the database, they only store a value in the gender field if the artist is female
# otherwise it is null
def convertGender(s):
    if s == 'male':
        return 0
    else:
        return 1
# register user defined functions
convertGenderUDF = udf(lambda x:convertGender(x), IntegerType())

# take artistGender column, apply user-defined function to columns and put it in new column 
def genderColumn(df):
    # change nulls to male here because I don't know what datatype spark uses to represent nulls
    # So I cant detect if a value is null when I'm applying convertGender to each individual value in the column
    x = df.na.fill(value='male', subset=['artistGender'])
    # create new column that is made from applying the custom function to the gender column
    x = x.withColumn("isFemale", convertGenderUDF(col("artistGender")))
    x = x.drop("artistGender")
    return x

def vectorizeNationality(df):
    # change strings in nationality column to list of tokens, which is just each word in the string
    nationalityTokenizer = Tokenizer(outputCol="nationalityWords")
    nationalityTokenizer.setInputCol("artistNationality")
    df = nationalityTokenizer.transform(df)

    # word2vec doesn't really extract any interesting patterns from the nationality column
    # but it does help in reducing its dimensions
    # in the bag of words approach, there were 108 dimensions
    word2vec = Word2Vec(vectorSize=5, inputCol="nationalityWords", outputCol='nationalityVec', seed=123)
    model = word2vec.fit(df)
    df = model.transform(df)
    df = df.drop('artistNationality')
    df = df.drop('nationalityWords')
    return df

#exact same process as vectorizeNationality
def vectorizeObject(df):
    objectNameTokenizer = Tokenizer(outputCol="objectNameWords")
    objectNameTokenizer.setInputCol("objectName")
    df = objectNameTokenizer.transform(df)

    word2vecObject = Word2Vec(vectorSize=5, inputCol="objectNameWords", outputCol='objectNameVec', seed=456)
    model = word2vecObject.fit(df)
    df = model.transform(df)
    
    df = df.drop('objectName')
    df = df.drop('objectNameWords')

    return df

def oneHotDepartment(df):
    stringid = StringIndexer(inputCol='department', outputCol='departmentIndex')
    model = stringid.fit(df)
    df = model.transform(df)

    ohe = OneHotEncoder(inputCol='departmentIndex', outputCol='departmentCode')
    model2 = ohe.fit(df)
    df = model2.transform(df)
    
    df = df.drop('department')
    df = df.drop('departmentIndex')
    

    return df

# transforms the dataset into the form we need to train the model
def transformIntoInput(df, weightVals:dict=None):
    features = df.columns
    features.remove('isHighlightInt')
    vector = VectorAssembler(inputCols=features, outputCol='features', handleInvalid='skip')
    vec = vector.transform(df)
    vec = vec.select(col("isHighlightInt").alias("label"), col("features"))

    if weightVals != None:
        vec = vec.withColumn('weight', when(vec.label == 1, weightVals[1]).otherwise(weightVals[0]))
    return vec

def makeConfusionMatrix(modelVector):
    # get positive predictions
      positivePredictions = modelVector.filter(modelVector.prediction == 1.)
      #true positives and false positives
      truePositives = positivePredictions.filter(positivePredictions.label == 1.).count()
      falsePositives = positivePredictions.filter(positivePredictions.label == 0.).count()
      # get all negative predictions
      negativePredictions = modelVector.filter(modelVector.prediction == 0.)
      # true negatives and false negatives
      trueNegatives = negativePredictions.filter(negativePredictions.label == 0.).count()
      falseNegatives = negativePredictions.filter(negativePredictions.label == 1.).count()
      # print the confusion matrix
      print("Confusion Matrix:\n\
            \t\t\tpositive prediction\tnegative prediction\n\
            positive label\t\t\t{tp}\t{fn}\n\
            negative label\t\t\t{fp}\t{tn}"\
            .format(tp = truePositives, fn = falseNegatives, fp = falsePositives, tn = trueNegatives))
      return [[truePositives, falseNegatives], [falsePositives, trueNegatives]]

if __name__ == "__main__":
    spark = SparkSession.builder.master("local[4]").appName("Predict").config("spark.ui.port", '4050').getOrCreate()
    sc = spark.sparkContext
    # get csv file that was created in the extract file
    # extract program is set up so that only one csv will be in the extractedDataset folder, so we can use this regex to
    # get the right file without having to type the whole name out

    # You could change this path to be any csv file as long as is has the same schema as the extracted dataset
    fileName = subprocess.check_output('ls extractedDataset | grep ".*.csv"', shell=True, text=True).removesuffix('\n')
    path = os.getcwd() + "/extractedDataset/" + fileName
    df = spark.read.csv(path, header=True, nullValue='')

    modelPath = os.getcwd() + "/final_model"
    model = LogisticRegressionModel.load(modelPath)
    df = transformDataset(df)
    df = transformIntoInput(df, weightVals={0: 1.0027375831052014, 1: 366.2857142857143})
    modelOutput = model.transform(df)

    matrix = makeConfusionMatrix(modelOutput)
    print("\npredictions:\n")
    print(modelOutput.show())

    spark.stop()