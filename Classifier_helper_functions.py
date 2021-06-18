# imports
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


# load large data set
sdc = pd.read_csv('data/stroke_data_complete.csv')
# fill missing data with Pandas
sdc["RATRIAL"].fillna(value='N', inplace=True)
# convert Y/N/C to 1/0/0
cats = ['DDIAGISC', 'DDIAGHA', 'DDIAGUN', 'DNOSTRK', 'RSLEEP', 'RATRIAL', 'RVISINF', 'RDEF1', 'RDEF2', 'RDEF3', 'RDEF4',
        'RDEF5', 'RDEF6', 'RDEF7', 'RDEF8']
for cat in cats:
    sdc[cat] = sdc[cat].replace(to_replace=['N', 'Y', 'C'], value=[0, 1, 0])
# DROP predicted columns

sdc_forCLF = sdc.drop(columns=['STYPE', 'DDIAGISC', 'DDIAGHA', 'DDIAGUN', 'DNOSTRK'])


# Split Data

X = sdc_forCLF.drop('RVISINF', axis=1)
y = sdc_forCLF["RVISINF"]



categorical_features = ["mental_status", 'SEX']
one_hot = OneHotEncoder()
transformer = ColumnTransformer([("one_hot", one_hot, categorical_features)], remainder="passthrough")
transformed_sdc = transformer.fit_transform(X)

# Split and train data
X_train, X_test, y_train, y_test = train_test_split(transformed_sdc, y, test_size=.2)

# load ML model
clf = joblib.load('sx_clf_rf.pkl')

print(X_test[:1])








#strokeModel.fit(X_train, y_train);
#training_score = strokeModel.score(X_train, y_train)
#prediction_score = strokeModel.score(X_test, y_test)


# perform different tests

def modelHealthScores(model, td_score, testd_score):
    print(f'training score: {td_score} prediction_score:{testd_score}')
    # crossval score
    crossValScore = cross_val_score(model, transformed_X, y, cv=5)
    print(crossValScore)
    return td_score, testd_score, crossValScore


# creates the confusion matrix using y test values and y prediction values
def create_confusion(model, x_data):
    y_preds = model.predict(x_data)
    stroke_confusion_matrix = confusion_matrix(y_test, y_preds)
    return stroke_confusion_matrix


def plot_conf_mat(conf_mat):
    fig, ax = plt.subplots(figsize=(3, 3))
    ax = sns.heatmap(conf_mat, annot=True, cbar=False)
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.savefig('static/conf_mat.png', dpi=400)


def save_classificaiton_report():
    y_preds = strokeModel.predict(X_test)
    pd.DataFrame(classification_report(y_test, y_preds, output_dict=True, zero_division=True)).to_pickle(
        "../../Desktop/WGU/strokeAI/data/classification_report.pkl")
    print("file saved to static folder")

# save_classificaiton_report()

# stroke_conf_mat = create_confusion(strokeModel, X_test)
# conf_mat_vis=plot_conf_mat(stroke_conf_mat).savefig('confusion_matrix.png',dpi=400)


# modelHealthScores(strokeModel, training_score, prediction_score)
