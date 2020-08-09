# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 22:05:39 2020

@author: subhrohalder
"""
#dataset link: https://www.kaggle.com/orgesleka/used-cars-database
#Importing the libraries
import pandas as pd
import numpy as np
import seaborn as sns

#importing the dataset
df=pd.read_csv('autos.csv', encoding = "ISO-8859-1")


#Data cleaning processs
#checking for null values
df.isnull().sum()
df_new = df.dropna()
df_new.isnull().sum()

#feature selection
for col in df_new:
    print(col)
    print(df_new[col].unique())
    print(len(df_new[col].unique()))

    
df_new['seller'].value_counts()
# =============================================================================
# privat        260963
# gewerblich         2
#will drop the seller column as we can see it has only two data for gewerblich
# =============================================================================


df_new['name'].value_counts()
#name columns have 150359 different catergory and name of the car is not important


df_new['offerType'].value_counts()
# =============================================================================
# Angebot    260961
# Gesuch          4
#will drop this column as well as it has only four values of Gesuch
# =============================================================================

df_new['nrOfPictures'].value_counts()
# it has only one value 0.0 therefore it is not a important feature and will drop it

df_new['abtest'].value_counts()


df_new['powerPS'].value_counts()


#droping least important features
df_new=df_new.drop(['dateCrawled','name','seller','offerType',
                                'nrOfPictures','postalCode','lastSeen'
                                ,'dateCreated'],axis='columns')
    

#checking categorical feature
print('vehicle_type: ',df_new.vehicleType.unique())
print('gearbox: ',df_new.gearbox.unique())
print('fuel_type: ',df_new.fuelType.unique())
print('repaired_damage: ',df_new.notRepairedDamage.unique())

df_new.shape


#Removing the outliers
df_new.kilometer = df_new.kilometer.astype('int64')

sns.boxplot(x=df_new['yearOfRegistration'])
sns.boxplot(x=df_new['price'])
sns.boxplot(x=df_new['kilometer'])
sns.boxplot(x=df_new['powerPS'])

df_new = df_new.loc[(df_new.price>400)&(df_new.price<=40000)]
df_new = df_new.loc[(df_new.yearOfRegistration>1990)&(df_new.yearOfRegistration<=2019)]
df_new = df_new.loc[(df_new.powerPS>10)]
df_new = df_new.loc[(df_new.kilometer>1000)&(df_new.kilometer<=150000)]

df_new.monthOfRegistration.value_counts()
# there are 7603 0.0 changing it to 1.0
df_new.monthOfRegistration.replace(0,1,inplace=True)

#converting to date
register_date = pd.to_datetime(df_new.yearOfRegistration*10000+df_new.monthOfRegistration*100+1,format='%Y%m%d')

#current date
from datetime import date
current_date=date(2020, 8,9)

#no of days old
old_days=(pd.to_datetime(current_date) - register_date)
#formmating the days
old_days=(old_days / np.timedelta64(1, 'D')).astype(int)

#adding it to dataframe
df_new['old_days'] = old_days

#droping as we have no of days old now
df_new.drop(columns=['yearOfRegistration','monthOfRegistration'],inplace=True)

#Converting categorical variable to numerical variable
df_dummies=pd.get_dummies(data=df_new,columns=['notRepairedDamage','vehicleType','model','brand','gearbox','fuelType','abtest'],drop_first=True)

#independent variable
X = df_dummies.drop('price',axis=1)

#dependent variable
y = df_dummies.price

#train test split
from sklearn.model_selection import train_test_split
from sklearn import metrics
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.20, random_state=0)

#model building
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

#MSE
np.sqrt(metrics.mean_squared_error(y_test, y_pred))
#3288.651275504349

#Predicting the test set results
y_pred = model.predict(X_test)
print(model.score(X_test, y_test)*100,'% Prediction Accuracy')
#75.95798025286301 % Prediction Accuracy

















