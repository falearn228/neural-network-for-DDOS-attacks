import numpy as np
import pandas as pd
# %matplotlib inline
import sklearn as skl
import time
seconds = time.time()
from sklearn.ensemble import RandomForestClassifier

dtypes = {'Src IP': 'category', 'Src Port': 'uint16', 'Dst IP': 'category', 'Dst Port': 'uint16', 'Protocol': 'float32', 'Flow Duration': 'uint32', 'Tot Fwd Pkts': 'uint32', 'Tot Bwd Pkts': 'uint32', 'TotLen Fwd Pkts': 'float32', 'TotLen Bwd Pkts': 'float32', 'Fwd Pkt Len Max': 'float32', 'Fwd Pkt Len Min': 'float32', 'Fwd Pkt Len Mean': 'float32', 'Fwd Pkt Len Std': 'float32', 'Bwd Pkt Len Max': 'float32', 'Bwd Pkt Len Min': 'float32', 'Bwd Pkt Len Mean': 'float32', 'Bwd Pkt Len Std': 'float32', 'Flow Byts/s': 'float32', 'Flow Pkts/s': 'float32', 'Flow IAT Mean': 'float32', 'Flow IAT Std': 'float32', 'Flow IAT Max': 'float32', 'Flow IAT Min': 'float32', 'Fwd IAT Tot': 'float32', 'Fwd IAT Mean': 'float32', 'Fwd IAT Std': 'float32', 'Fwd IAT Max': 'float32', 'Fwd IAT Min': 'float32', 'Bwd IAT Tot': 'float32', 'Bwd IAT Mean': 'float32', 'Bwd IAT Std': 'float32', 'Bwd IAT Max': 'float32', 'Bwd IAT Min': 'float32', 'Fwd PSH Flags': 'float32', 'Bwd PSH Flags': 'float32', 'Fwd URG Flags': 'float32', 'Bwd URG Flags': 'float32', 'Fwd Header Len': 'uint32', 'Bwd Header Len': 'uint32', 'Fwd Pkts/s': 'float32', 'Bwd Pkts/s': 'float32', 'Pkt Len Min': 'float32', 'Pkt Len Max': 'float32', 'Pkt Len Mean': 'float32', 'Pkt Len Std': 'float32', 'Pkt Len Var': 'float32', 'FIN Flag Cnt': 'float32', 'SYN Flag Cnt': 'float32', 'RST Flag Cnt': 'float32', 'PSH Flag Cnt': 'float32', 'ACK Flag Cnt': 'float32', 'URG Flag Cnt': 'float32', 'CWE Flag Count': 'float32', 'ECE Flag Cnt': 'float32', 'Down/Up Ratio': 'float32', 'Pkt Size Avg': 'float32', 'Fwd Seg Size Avg': 'float32', 'Bwd Seg Size Avg': 'float32', 'Fwd Byts/b Avg': 'uint32', 'Fwd Pkts/b Avg': 'uint32', 'Fwd Blk Rate Avg': 'uint32', 'Bwd Byts/b Avg': 'uint32', 'Bwd Pkts/b Avg': 'uint32', 'Bwd Blk Rate Avg': 'uint32', 'Subflow Fwd Pkts': 'uint32', 'Subflow Fwd Byts': 'uint32', 'Subflow Bwd Pkts': 'uint32', 'Subflow Bwd Byts': 'uint32', 'Init Fwd Win Byts': 'uint32', 'Init Bwd Win Byts': 'uint32', 'Fwd Act Data Pkts': 'uint32', 'Fwd Seg Size Min': 'uint32', 'Active Mean': 'float32', 'Active Std': 'float32', 'Active Max': 'float32', 'Active Min': 'float32', 'Idle Mean': 'float32', 'Idle Std': 'float32', 'Idle Max': 'float32', 'Idle Min': 'float32', 'Label': 'category'}
ths = open("importance_list_all_data.csv", "w")
folder = ("./feaure_pics/")
df = pd.read_csv("final_dataset.csv", dtype = dtypes, usecols= [*dtypes.keys()], engine = 'c', low_memory= True, nrows = 100)
print(df.describe(include='all'))
columnsToDrop = np.array(['Fwd Byts/b Avg', 'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg', 'Bwd Byts/b Avg', 'Bwd Pkts/b Avg', 'Bwd Blk Rate Avg'])  # Убираем колонки , где лишь одно значение
col = np.union1d(columnsToDrop, ['Src IP', 'Dst IP'])
df.shape   # 12794627 83
df.drop(columns = col, inplace = True)
df.replace([np.inf, -np.inf], np.nan, inplace=True) # Убираем бесконечно большие и малые числа , заменив их на значения NaN
df = df.fillna(0)      # Убираем строки со значением NaN
df['Label'].replace(['Benign','ddos'],[1,0],inplace=True)         # Превращение выходных значений в бинарный вид

y = df["Label"].values
del df["Label"]
X = df.values

forest = skl.ensemble.RandomForestClassifier(n_estimators=250,random_state=0)
forest.fit(X, y)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],axis=0)

indices = np.argsort(importances)[::-1]
refclasscol=list(df.columns.values)

impor_bars = pd.DataFrame({'Features':refclasscol[0:20],'importance':importances[0:20]})
impor_bars = impor_bars.sort_values('importance',ascending=False).set_index('Features')
#printing the feature importances
count=0
fea_ture="final_dataset"+"=["
for i in impor_bars.index:
    fea_ture=fea_ture+"\""+str(i)+"\","
    count+=1
    if count==5:
        fea_ture=fea_ture[0:-1]+"]"
        break
print("final_dataset", "importance list:")
print("final_dataset","\n",impor_bars.head(20),"\n\n\n")
print(fea_ture)
print("-----------------------------------------------------------------------------------------------\n\n\n\n")

print("Конец....")
print("Потраченное количество времени: = ", time.time()-seconds, "секунд")
ths.close()


#дисперсия отклонения обратного пакета