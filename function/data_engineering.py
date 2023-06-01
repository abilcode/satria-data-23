import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import f1_score, accuracy_score, classification_report

def cross_validation(X,y,model):
    f1 = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        f1_macro = f1_score(y_test, y_pred, average = 'macro')
        f1.append(f1_macro)
        print(classification_report(y_test, y_pred),np.mean(f1_macro))
    return f1

def freq_encoding(data, cat_feat):
  grouped_data = data.groupby([cat_feat]).size()/data.shape[0]
  data.loc[:, f'{cat_feat}_encode'] = data[cat_feat].map(grouped_data)
  return data

def binding(data,cols): 
    for i in cols:
        for c in cols:
            data[f'{c}_nominal'] = pd.cut(data[c], bins=5, labels=[1,2,3,4,5])