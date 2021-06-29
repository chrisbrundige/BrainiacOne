# imports
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# IMPORT DATA SET AND USE ONLY 500 values
sx_dx = pd.read_csv('data/stroke_data_complete.csv')
#sx_dx = sx_dx[2000:2500]

# convert data to binary
cats = ['DDIAGISC', 'DDIAGHA', 'DDIAGUN', 'DNOSTRK', 'RSLEEP', 'RATRIAL', 'RVISINF', 'RDEF1', 'RDEF2', 'RDEF3',
        'RDEF4', 'RDEF5', 'RDEF6', 'RDEF7', 'RDEF8']
for cat in cats:
    sx_dx[cat] = sx_dx[cat].replace(to_replace=['N', 'Y', 'C', 'u', 'U'], value=[0, 1, 0, 0, 0])

# create a new Feature that represents all strokes as a comination of isc. and hem. cva
# this has to happen after features have been converted to bin.
sx_dx['total_strokes'] = sx_dx['DDIAGISC'] + sx_dx['DDIAGHA']

# FORMAT NAN
pd.DataFrame.sx_dx=sx_dx.RATRIAL.fillna(sx_dx.RATRIAL.dropna(), inplace=True)
print(sx_dx.isna().sum())





# drop cats
sx_dx = sx_dx.drop(columns=['DDIAGISC', 'DDIAGHA', 'DDIAGUN', 'DNOSTRK', 'STYPE'])

# Split into X,y
X = sx_dx.drop('total_strokes', axis=1)
y = sx_dx["total_strokes"]

# convert other features to bin
categorical_features = ["mental_status", 'SEX']
one_hot = OneHotEncoder()
transformer = ColumnTransformer([("one_hot", one_hot, categorical_features)], remainder="passthrough")
trans_X = transformer.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(trans_X, y, test_size=.2)

clf = joblib.load('sx_clf_rf.pkl')
clf.fit(X_train, y_train)
y_preds = clf.predict(X_test)


def getScore(X,y):
    return clf.score(X,y, sample_weight=None)
    return model_score


def get_clf_report(ytest,ypred):
    return classification_report(ytest,ypred)


def get_conf_mat(ytest,ypred):
    return confusion_matrix(ytest,ypred)



print(getScore(X_train, y_train))

print(get_clf_report(y_test,y_preds))

print(get_conf_mat(y_test,y_preds))
