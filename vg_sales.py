import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from collections import Counter
from sklearn import datasets,ensemble

data_name = r'D:\研一下学期\数据挖掘\数据挖掘3\vgsales.csv'
file = pd.read_csv(data_name)
print(file.columns)
file.dropna(inplace=True)
file.drop(columns="Rank",inplace=True)
file = file[file["Year"]<2017.0]
file.describe()

platform_list = []
for item in file.loc[:,'Platform']:
        if item not in platform_list:
                platform_list.append(item)
print(platform_list)

year_list = []
for item in file.loc[:,'Year']:
        if item not in year_list and not math.isnan(item):
                year_list.append(item)
year_list.sort()            
print(year_list)

genre_list = []
for item in file.loc[:,'Genre']:
        if item not in genre_list:
                genre_list.append(item)            
print(genre_list)

pub_list = []
for item in file.loc[:,'Publisher']:
        if item not in pub_list:
                pub_list.append(item)            
print(pub_list[:10])

sale_att = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']
for att in sale_att:
        for item in file.loc[:,att]:
            if math.isnan(item):
                    print(att+'nan')
                    break
                    
#Visualization
sale_year=file[['Year', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']].groupby('Year').sum()
sale_year.index=sale_year.index.astype(int)
sale_year.plot.bar(figsize=(20,10),fontsize=30)
plt.tight_layout()
plt.show()

plt.figure(figsize=(30,10))
plt.bar(file[['Name','Global_Sales']].sort_values('Global_Sales',ascending=False)[:10]['Name'], file[['Name','Global_Sales']].sort_values('Global_Sales',ascending=False)[:10]['Global_Sales'])
plt.show()

genre_sale = file[['Genre','Global_Sales']].groupby('Genre').sum()
genre_sale.plot.bar(figsize=(20,10),fontsize=30)
plt.tight_layout()
plt.show()

import seaborn as sns
sales=file[['Genre', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']]
sales = sales.groupby('Genre').sum()
plt.figure(figsize=(20,10))
sns.heatmap(sales,annot=True,fmt= '.0f')

plt.figure(figsize=(30,10))
plt.bar(file['Platform'].value_counts().index,file['Platform'].value_counts())

genre_sale = file[['Platform','Global_Sales']].groupby('Platform').sum()
genre_sale.plot.bar(figsize=(20,10),fontsize=30)
plt.tight_layout()
plt.show()

plt.figure(figsize=(30,10))
plt.hist(file['Year'],bins=[a for a in range(1981,2021)],rwidth=0.5)
plt.show()

#Clustering Analization
from pyclust import KMedoids
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

file = pd.read_csv(data_name, keep_default_na=False)
file.dropna(inplace=True)
file.drop(columns="Rank",inplace=True)
sales=file[[ 'Genre','NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']]
print(sales)
datamat=sales.values.tolist()
datamat = np.array(datamat)

X = file.iloc[:, 5:9].values[:5000]
xx=file.iloc[:,4].values[:5000]
Y = file.iloc[:, 3].values[:5000]
print(xx)
print(X)
print(Y)
p_dict = {}
p_num=0
for idx,i in enumerate(xx):
    if i not in p_dict.keys():
            p_dict[i]=p_num
            p_num+=1
    xx[idx]=p_dict[i]
X_np = np.array(X)
xx_np = np.array(xx)
print(xx_np)
a=np.insert(X_np,0,values=xx_np,axis=1)
print(a)

#K-Means
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 12)
kmeans.fit(X)
kmeans.cluster_centers_
distance = kmeans.fit_transform(X)
distance
labels = kmeans.labels_
labels
plt.scatter(X[:, 0], X[:,1], s = 100, c = kmeans.labels_)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 50, c = 'red',label = 'Centroids')
plt.title('Jogos Clusters and Centroids')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()

#KNN Method
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)

from sklearn.preprocessing import StandardScaler  
scaler = StandardScaler()  
scaler.fit(X_train)

X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier  
classifier = KNeighborsClassifier(n_neighbors=50)  
classifier.fit(X_train, y_train)  

y_pred = classifier.predict(X_test)  

from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))  
 
#Game Sales Prediction
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

file = pd.read_csv(data_name, keep_default_na=False)
file = file.dropna(axis=0, subset=['Year','Publisher'])
x = file.iloc[:,6:-1].values
y = file.iloc[:,-1].values

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
forestRange=range(50,500,50)
scores_list=[]
for i in forestRange: 
    regressor_Forest = RandomForestRegressor(n_estimators=i,random_state=0)
    regressor_Forest.fit(x_train,y_train)
    y_pred = regressor_Forest.predict(x_test)
    scores_list.append(r2_score(y_test,y_pred))
plt.plot(forestRange,scores_list,linewidth=2,color='maroon')
plt.xticks(forestRange)
plt.xlabel('No. of trees')
plt.ylabel('r2 score of Random Forest Reg.')
plt.show() 
regressor_Forest = RandomForestRegressor(n_estimators=100,random_state=0)
regressor_Forest.fit(x_train,y_train)
y_pred = regressor_Forest.predict(x_test)
r2_forest = r2_score(y_test,y_pred)
print(r2_forest)

