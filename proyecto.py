# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline


#lectura del dataset train y test
train_df = pd.read_csv('./train.csv')
test_df = pd.read_csv('./test.csv')
#combine = [train_df, test_df]

#nombre de las variables
print(train_df.columns.values)

#descripcion del dataset
train_df.head()

#descripcion breve del dataset
train_df.tail()

#Transformacion de tipo dato category en lugar de object
print('_'*40)
train_df["Survived"]=train_df["Survived"].astype('category')
train_df["Pclass"]=train_df["Pclass"].astype('category')
train_df["Sex"]=train_df["Sex"].astype('category')
train_df["Cabin"]=train_df["Cabin"].astype('category')
train_df["Ticket"]=train_df["Ticket"].astype('category')
train_df["Embarked"]=train_df["Embarked"].astype('category')

#descripcion de los tipos de las variables
train_df.info()


#descripcion de los valores de las medidas numericas
train_df.describe(include='all')


#Obtener la media de la clase frente a supervivencia
train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)


#Obtener la media del sexo frente a supervivencia
train_df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)


#Obtener la media del sibling/spouse frente a supervivencia
train_df[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)

#Obtener la media del parent/child frente a supervivencia
train_df[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)

#visualizing data

#visualizacion de supervivencia frente edad
g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Age', bins=20)
plt.show()

#visualizacion de supervivencia frente edad
g = sns.FacetGrid(train_df, col='Pclass')
g.map(plt.hist, 'Age', bins=20)
plt.show()

#visualizacion de supervivencia frente clase y edad 
grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', height=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();
plt.show()

#visualizacion de supervivencia frente clase, edad y sex
grid = sns.FacetGrid(train_df, row='Sex', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', palette='deep')
grid.add_legend()
plt.show()

#visualizacion de supervivencia frente clase, edad , sex y ubicacion
grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()
plt.show()

#visualizacion de supervivencia frente embarcacion , fare, sexo
grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend()
plt.show()


grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend()
plt.show()