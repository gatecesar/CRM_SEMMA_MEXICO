from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler, Imputer, StandardScaler
from pyspark.sql.types import DoubleType
import pandas as pd
# Para selección de variables
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.classification import RandomForestClassifier
import random
from pyspark.ml import Pipeline
#Evaluacion
from pyspark.ml.evaluation import BinaryClassificationEvaluator

class Sample():
    """docstring for sample, it samples data."""

    def __init__(self, data, opt=None):
        self.data = data
        self.opt = opt


    def null_cleaner (self, cut_off = .3):
        validVars=[]
        dataCount = self.data.count()
        for col in self.data.columns:
                if (self.data.filter(self.data[col].isNull()).count()/dataCount < cut_off):
                    validVars.append(col)
        return self.data.select([item for item in self.data.columns if item in validVars])


    def imputation(self):
        C=[i[0] for i in self.data.dtypes if 'string' in i[1]]
        I=[i[0] for i in self.data.dtypes if 'int' in i[1]]

        for f in I: self.data = self.data.withColumn(f, self.data[f].cast(DoubleType()))
        imputer = Imputer(
    	    	      inputCols= [c for c in self.data.columns if c not in C],
    	              outputCols=[c for c in self.data.columns if c not in C])
        Pba=imputer.fit(self.data)
        return Pba.transform(self.data)


    def correlation(self , cut_off = .7 ):
    	Cor=[i[0] for i in self.data.dtypes if 'double' in i[1]]
    	C=[i[0] for i in self.data.dtypes if 'string' in i[1]]
    	vector_col = "corr_features"
    	assembler = VectorAssembler(inputCols= [c for c in Cor ],outputCol=vector_col)
    	df_vector = assembler.transform(self.data).select(vector_col)
    	matrix = Correlation.corr(df_vector, vector_col,"spearman").head()
    	a=matrix[0]
    	Arreglo=a.toArray()
    	CorrArrpd=pd.DataFrame(data=Arreglo[0:,0:])
    	CorrArrpd.columns = [c for c in Cor ]
    	col_corr= []
    	cut_off=.7
    	for i in range(len(CorrArrpd)):
    		for j in range(i):
    			if (CorrArrpd.iloc[i, j] >= cut_off) and (CorrArrpd.columns[j] not in col_corr):
    				colname = CorrArrpd.columns[i]
    				col_corr.append(colname)

    	CorrArrpd.drop(col_corr,inplace=True,axis=1)
    	sinCorrelacion=CorrArrpd.columns
    	joinedlist = list(set().union(sinCorrelacion,C))
    	self.data=self.data.select([sinCor for sinCor in self.data.columns if sinCor in joinedlist])
    	return self.data

    def VarSelection(self,Tgt='Target'):
    	Cor=[i[0] for i in self.data.dtypes if 'double' in i[1]]
    	vectorassembler = VectorAssembler(inputCols=[_ for _ in Cor if _ not in  (Tgt)],
    									  outputCol='assembled_features')
    	DataM = vectorassembler.transform(self.data)
    	random_seed = 4
    	num_iter = 10
    	random.seed(random_seed)
    	random_seeds=set([random.randint(0,10000) for _ in range(num_iter)])
    	features_random_seed = {}
    	for random_seed in random_seeds:
    		rf = RandomForestClassifier(featuresCol=vectorassembler.getOutputCol(), labelCol=Tgt, seed = random_seed)
    		rf_model = rf.fit(DataM)

    		importances = [(index, value) for index, value in enumerate(rf_model.featureImportances.toArray().tolist())]
    		importances = sorted(importances, key=lambda value: value[1], reverse=True)
    		imp = 0
    		vector_assembler_cols = vectorassembler.getInputCols()
    		for element in importances:
    			feature = vector_assembler_cols[element[0]]
    			importance = element[1 

    			if imp < 0.95:
    				features_random_seed[feature] = features_random_seed.get(feature, []) + [importance]
    			else:
    				features_random_seed[feature] = features_random_seed.get(feature, []) + [None]
    			imp += element[1]
    	features_random_seed = pd.DataFrame(features_random_seed).T
    	feature_importances = features_random_seed.dropna(how='all').mean(axis=1)
    	list_of_feature_importance = sorted(zip(feature_importances.index, feature_importances),
    							key=lambda x: x[1],
    							reverse=True)
    	print(list_of_feature_importance)
    	return list_of_feature_importance
