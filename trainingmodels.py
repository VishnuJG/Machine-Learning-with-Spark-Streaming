from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer
from pyspark.ml.feature import StopWordsRemover
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.feature import HashingTF
from pyspark.ml.feature import Tokenizer
from pyspark.ml.feature import StopWordsRemover
from sklearn.naive_bayes import MultinomialNB
from sklearn.cluster import MiniBatchKMeans
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.feature import HashingTF
from pyspark.ml.feature import NGram
from pyspark.ml.feature import StringIndexer
from sklearn.linear_model import SGDClassifier
from pyspark.ml import Pipeline
from sklearn.naive_bayes import BernoulliNB
from pyspark.ml.pipeline import PipelineModel
import joblib
import numpy as np
import json

sc = SparkContext("local[2]", "spam-ham")
ssc = StreamingContext(sc, 1)
spark=SparkSession(sc)
sqlc=SQLContext(sc)
modelspath="/models"
lines = ssc.socketTextStream("localhost", 6100)

def fun(blocks):
    if blocks.isEmpty():
        return
    else:
        res=blocks.collect()
        for i in res:
            eachrow=json.loads(i, strict=False)
        newrow=[]
        for i in eachrow.keys():
            temp=[]
            temp.append(str(eachrow[i]['feature0']).strip(' '))
            temp.append(str(eachrow[i]['feature1']).strip(' '))
            temp.append(str(eachrow[i]['feature2']).strip(' '))
            temp.append(len(str(eachrow[i]['feature1'])))
            newrow.append(temp)
        df=preprocessingfun(newrow,sc)
        
def preprocessingfun(df,sc):
    spark=SparkSession(sc)
    dff=spark.createDataFrame(df, schema="dfsubject string, dfmessage string, hamorspam string, lengthmessage long")
    tokenizedcol=Tokenizer(inputCol="dfmessage", outputCol="tokenizedtext")
    stopwords = StopWordsRemover().getStopWords() + ['-']
    stopwordsremoved = StopWordsRemover().setStopWords(stopwords).setInputCol('tokenizedtext').setOutputCol('stopwordsremoved')
    bigchunks = NGram().setN(2).setInputCol('stopwordsremoved').setOutputCol('bigchunks')
    ht = HashingTF(inputCol="bigchunks", outputCol="hastb",numFeatures=8000)
    labelizing = StringIndexer(inputCol='hamorspam',outputCol='targetcol')
    dataprepropipe=Pipeline(stages=[tokenizedcol, stopwordsremoved, bigchunks, ht, labelizing])
    cleanerdf = dataprepropipe.fit(dff)
    cleandf = cleanerdf.transform(dff)
    cleandf = cleandf.select(['targetcol','stopwordsremoved','hastb','bigchunks'])
    #cleandf.show()
    #bernoullimodelfit(cleandf)
    #linearClassifier(cleandf)
    #naiveBayesian(cleandf)
    clusteringkmeans(cleandf)
    return
    
    
def bernoullimodelfit(cleandf):
    dependentvar = np.array(cleandf.select('hastb').collect())
    independentvar=np.array(cleandf.select('targetcol').collect())
    nsamples, nx, ny = dependentvar.shape
    dependentvar = dependentvar.reshape((nsamples,nx*ny))
    
    try:
        modelload = joblib.load('/home/pes1ug19cs574/project/models/bernoullimodel.pkl')
        modelload.partial_fit(dependentvar, independentvar.ravel())
        joblib.dump(modelload, '/home/pes1ug19cs574/project/models/bernoullimodel.pkl')
    except Exception as e:
        print("**************************************")
        newmodel=BernoulliNB()
        newmodel.partial_fit(dependentvar, independentvar.ravel(),classes=np.unique(independentvar))
        joblib.dump(newmodel, '/home/pes1ug19cs574/project/models/bernoullimodel.pkl')
    return 


def linearClassifier(cleandf):#SGD is stochastic gradient descent --- this is for reference has nothing to do with the code
    dependentvar = np.array(cleandf.select('hastb').collect())
    independentvar=np.array(cleandf.select('targetcol').collect())
    nsamples, nx, ny = dependentvar.shape
    dependentvar = dependentvar.reshape((nsamples,nx*ny))
    try:
        modelload = joblib.load('/home/pes1ug19cs574/project/models/SGD.pkl')
        modelload.partial_fit(dependentvar, independentvar.ravel())
        joblib.dump(modelload, '/home/pes1ug19cs574/project/models/SGD.pkl')
    except Exception as e:
        print("**************************************")
        newmodel=SGDClassifier()
        newmodel.partial_fit(dependentvar, independentvar.ravel(),classes=np.unique(independentvar))
        joblib.dump(newmodel, '/home/pes1ug19cs574/project/models/SGD.pkl')
    return 

def naiveBayesian(cleandf):
    
    dependentvar = np.array(cleandf.select('hastb').collect())
    independentvar=np.array(cleandf.select('targetcol').collect())
    nsamples, nx, ny = dependentvar.shape
    dependentvar = dependentvar.reshape((nsamples,nx*ny))
    try:
        modelload = joblib.load('/home/pes1ug19cs574/project/models/naive.pkl')
        modelload.partial_fit(dependentvar, independentvar.ravel())
        joblib.dump(modelload, '/home/pes1ug19cs574/project/models/naive.pkl')
    except Exception as e:
        print("**************************************")
        newmodel=MultinomialNB()
        newmodel.partial_fit(dependentvar, independentvar.ravel(),classes=np.unique(independentvar))
        joblib.dump(newmodel, '/home/pes1ug19cs574/project/models/naive.pkl')
    return 

def clusteringkmeans(cleandf):
    
    dependentvar = np.array(cleandf.select('hastb').collect())
    independentvar=np.array(cleandf.select('targetcol').collect())
    nsamples, nx, ny = dependentvar.shape
    dependentvar = dependentvar.reshape((nsamples,nx*ny))
    try:
        modelload = joblib.load('/home/pes1ug19cs574/project/models/kmean.pkl')
        modelload.partial_fit(dependentvar, independentvar.ravel())
        joblib.dump(modelload, '/home/pes1ug19cs574/project/models/kmean.pkl')
    except Exception as e:
        print("**************************************")
        newmodel=MiniBatchKMeans(n_clusters=2,random_state=0)
        newmodel.partial_fit(dependentvar, independentvar.ravel())
        joblib.dump(newmodel, '/home/pes1ug19cs574/project/models/kmean.pkl')
    return 
block=lines.flatMap(lambda line:line.split("\n"));

block.foreachRDD(lambda x:fun(x))


#block.pprint()




ssc.start()             
ssc.awaitTermination() 

