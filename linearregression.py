import numpy as np

import matplotlib.pyplot as mp

import seaborn as sb

import pandas as pd

from google.colab import files

u=files.upload()

dataset=pd.read_csv('50_Startups.csv')

x=dataset.iloc[:,:-1]

x

y=dataset.iloc[:,4]

y

from sklearn.preprocessing import LabelEncoder,OneHotEncoder

label=LabelEncoder()

x.iloc[:,3]=label.fit_transform(x.iloc[:,3])

one=OneHotEncoder(categorical_features=[3])

x=one.fit_transform(x).toarray()

x

x=x[:,1:]

x

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

x

y

x_train

y_test

from sklearn.linear_model import LinearRegression

r=LinearRegression()

r.fit(x_train,y_train)

y_pred=r.predict(x_test)

y_pred

mp.scatter(x_test[:,0],y_test,color='green')
mp.plot(x_test[:,0],y_pred,color='blue')




