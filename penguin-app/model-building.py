import pandas as pd
import sklearn
penguins = pd.read_csv('penguins_cleaned.csv')
df = penguins.copy()
target = 'species'
encode = ['sex','island']
df = pd.get_dummies(df,columns=encode,prefix = encode)
target_mapper = {'Adelie':0,'Chinstrap':1,'Gentoo':2}
def target_encode(val):
    return target_mapper[val]

df['species'] = df['species'].apply(target_encode)

X = df.drop('species',axis =1)
Y =df['species']

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X,Y)
import pickle
pickle.dump(clf,open('penguins_clf.pkl','wb'))
