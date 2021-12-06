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
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.feature import HashingTF
from pyspark.ml.feature import NGram
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import r2_score,accuracy_score, precision_score, recall_score
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
    clusteringpred(cleandf)
    
def bernoullimodelpred(cleandf):
    dependentvar = np.array(cleandf.select('hastb').collect())
    independentvar=np.array(cleandf.select('targetcol').collect())
    nsamples, nx, ny = dependentvar.shape
    dependentvar = dependentvar.reshape((nsamples,nx*ny))
    
    try:
        modelload = joblib.load('/home/pes1ug19cs574/project/models/bernoullimodel.pkl')
        predres=modelload.predict(dependentvar)
        #print(predres)
        
        acc = accuracy_score(independentvar, predres)
        pr = precision_score(independentvar, predres)
        f = open("burnacc.txt", "a")
        f.write("Accuracy : "+str(acc)+" Precision : "+str(pr)+"\n")
        f.close()
        print("the accuacy is ",acc)
        print("the precision: ",pr)
        
        
    except Exception as e:
        print(e)
        #newmodel=BernoulliNB()
        #newmodel.partial_fit(dependentvar, independentvar.ravel(),classes=np.unique(independentvar))
        #joblib.dump(newmodel, '/home/pes1ug19cs574/project/models/bernoullimodel.pkl')
    return 
    
def linearmodelpred(cleandf):
    dependentvar = np.array(cleandf.select('hastb').collect())
    independentvar=np.array(cleandf.select('targetcol').collect())
    nsamples, nx, ny = dependentvar.shape
    dependentvar = dependentvar.reshape((nsamples,nx*ny))
    
    try:
        modelload = joblib.load('/home/pes1ug19cs574/project/models/SGD.pkl')
        predres=modelload.predict(dependentvar)
        #print(predres)
        
        acc = accuracy_score(independentvar, predres)
        pr = precision_score(independentvar, predres)
        f = open("SGD.txt", "a")
        f.write("Accuracy : "+str(acc)+" Precision : "+str(pr)+"\n")
        f.close()
        print("the accuacy is ",acc)
        print("the precision: ",pr)
        
        
    except Exception as e:
        print(e)
        #newmodel=BernoulliNB()
        #newmodel.partial_fit(dependentvar, independentvar.ravel(),classes=np.unique(independentvar))
        #joblib.dump(newmodel, '/home/pes1ug19cs574/project/models/bernoullimodel.pkl')
    return 
def naivebayesianpred(cleandf):
    dependentvar = np.array(cleandf.select('hastb').collect())
    independentvar=np.array(cleandf.select('targetcol').collect())
    nsamples, nx, ny = dependentvar.shape
    dependentvar = dependentvar.reshape((nsamples,nx*ny))
    
    try:
        modelload = joblib.load('/home/pes1ug19cs574/project/models/naive.pkl')
        predres=modelload.predict(dependentvar)
        #print(predres)
        
        acc = accuracy_score(independentvar, predres)
        pr = precision_score(independentvar, predres)
        f = open("naive.txt", "a")
        f.write("Accuracy : "+str(acc)+" Precision : "+str(pr)+"\n")
        f.close()
        print("the accuacy is ",acc)
        print("the precision: ",pr)
        
        
    except Exception as e:
        print(e)
        #newmodel=BernoulliNB()
        #newmodel.partial_fit(dependentvar, independentvar.ravel(),classes=np.unique(independentvar))
        #joblib.dump(newmodel, '/home/pes1ug19cs574/project/models/bernoullimodel.pkl')
    return 
def clusteringpred(cleandf):
    dependentvar = np.array(cleandf.select('hastb').collect())
    independentvar=np.array(cleandf.select('targetcol').collect())
    nsamples, nx, ny = dependentvar.shape
    dependentvar = dependentvar.reshape((nsamples,nx*ny))
    
    try:
        modelload = joblib.load('/home/pes1ug19cs574/project/models/kmean.pkl')
        predres=modelload.predict(dependentvar)
        #print(predres)
        
        acc = accuracy_score(independentvar, predres)
        pr = precision_score(independentvar, predres)
        f = open("kmeans.txt", "a")
        f.write("Accuracy : "+str(acc)+" Precision : "+str(pr)+"\n")
        f.close()
        print("the accuacy is ",acc)
        print("the precision: ",pr)
        
        
    except Exception as e:
        print(e)
        #newmodel=BernoulliNB()
        #newmodel.partial_fit(dependentvar, independentvar.ravel(),classes=np.unique(independentvar))
        #joblib.dump(newmodel, '/home/pes1ug19cs574/project/models/bernoullimodel.pkl')
    return 
block=lines.flatMap(lambda line:line.split("\n"));

block.foreachRDD(lambda x:fun(x))


#block.pprint()




ssc.start()             
ssc.awaitTermination() 

