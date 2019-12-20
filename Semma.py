from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import Imputer 
from pyspark.sql.types import DoubleType
import pandas as pd
# Para selección de variables
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.classification import RandomForestClassifier
import random
from pyspark.ml.feature import StandardScaler
from pyspark.ml import Pipeline
#Evaluacion
from pyspark.ml.evaluation import BinaryClassificationEvaluator
def limpias (Datos, corte = .3):
    A=[]
    Nuloprueba = Datos.count()  
    for col in Datos.columns:
            if (Datos.filter(Datos[col].isNull()).count()/Nuloprueba < corte):
                A.append(col)
    return Datos.select([nuevos for nuevos in Datos.columns if nuevos in A])
def imputaciones(VarLimpias):
    C=[i[0] for i in VarLimpias.dtypes if 'string' in i[1]]
    I=[i[0] for i in VarLimpias.dtypes if 'int' in i[1]] 
	
    for f in I: VarLimpias = VarLimpias.withColumn(f, VarLimpias[f].cast(DoubleType()))
    imputer = Imputer(
	    	      inputCols= [c for c in VarLimpias.columns if c not in C],
	              outputCols=[c for c in VarLimpias.columns if c not in C])
    Pba=imputer.fit(VarLimpias)
    return Pba.transform(VarLimpias)
def correlacion(Finales , corte = .7):
	Cor=[i[0] for i in Finales.dtypes if 'double' in i[1]]
	C=[i[0] for i in Finales.dtypes if 'string' in i[1]]
	vector_col = "corr_features"
	assembler = VectorAssembler(inputCols= [c for c in Cor ],outputCol=vector_col)
	df_vector = assembler.transform(Finales).select(vector_col)
	matrix = Correlation.corr(df_vector, vector_col,"spearman").head()
	a=matrix[0]
	Arreglo=a.toArray()
	CorrArrpd=pd.DataFrame(data=Arreglo[0:,0:])
	CorrArrpd.columns = [c for c in Cor ]
	col_corr= []
	corte=.7
	for i in range(len(CorrArrpd)):
		for j in range(i):
			if (CorrArrpd.iloc[i, j] >= corte) and (CorrArrpd.columns[j] not in col_corr):
				colname = CorrArrpd.columns[i] 
				col_corr.append(colname)
				
	CorrArrpd.drop(col_corr,inplace=True,axis=1)
	sinCorrelacion=CorrArrpd.columns
	joinedlist = list(set().union(sinCorrelacion,C))  
	Finales=Finales.select([sinCor for sinCor in Finales.columns if sinCor in joinedlist])
	return Finales
def VarSelection(Data,Tgt='Target'):
	Cor=[i[0] for i in Data.dtypes if 'double' in i[1]]
	vectorassembler = VectorAssembler(inputCols=[_ for _ in Cor if _ not in  (Tgt)], 
									  outputCol='assembled_features')
	DataM = vectorassembler.transform(Data)
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
			importance = element[1]
			
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
def Model(Data,Tgt='Target',Indp='Nada'):
	
	vector_assembler = VectorAssembler(inputCols=Indp, outputCol='assembled_important_features')
	standard_scaler = StandardScaler(inputCol=vector_assembler.getOutputCol(), outputCol='standardized_features')
	rf = RandomForestClassifier(featuresCol=standard_scaler.getOutputCol(), labelCol=Tgt)
#	letters_train, letters_test = letters.randomSplit([0.8,0.2], seed=4)
	pipeline = Pipeline(stages=[vector_assembler, standard_scaler, rf])
	pipeline_model_rf = pipeline.fit(Data)
	return pipeline_model_rf
	
def Evaluate(Data,modelo,Tgt='Tgt'):	
	Evaluado = modelo.transform(Data)
	auc = BinaryClassificationEvaluator(rawPredictionCol='rawPrediction', labelCol=Tgt, metricName='areaUnderROC')
	return auc.evaluate(Evaluado)