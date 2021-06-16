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
stroke_dataset = pd.read_csv('../../Desktop/WGU/strokeAI/data/stroke_data.csv').drop(columns=["work_type", "Residence_type", "id", "ever_married"])
# convert to numbers
# fill missing data with Pandas
stroke_dataset["bmi"].fillna(stroke_dataset["bmi"].mean(), inplace=True)

# Split Data

X = stroke_dataset.drop("stroke", axis=1)
y = stroke_dataset["stroke"]

categorical_features = ["gender", "smoking_status"]
one_hot = OneHotEncoder()
transformer = ColumnTransformer([("one_hot", one_hot, categorical_features)], remainder="passthrough")

transformed_X = transformer.fit_transform(X)

# Split and train data
X_train, X_test, y_train, y_test = train_test_split(transformed_X, y, test_size=0.22)

# load ML model
strokeModel = joblib.load('strokePred.pkl')
strokeModel.fit(X_train, y_train);
training_score = strokeModel.score(X_train, y_train)
prediction_score = strokeModel.score(X_test, y_test)


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
    plt.savefig('static/conf_mat.png',dpi=400)


def save_classificaiton_report():
    y_preds = strokeModel.predict(X_test)
    pd.DataFrame(classification_report(y_test,y_preds,output_dict=True,zero_division=True)).to_pickle(
        "../../Desktop/WGU/strokeAI/data/classification_report.pkl")
    print("file saved to static folder")

save_classificaiton_report()









# stroke_conf_mat = create_confusion(strokeModel, X_test)
# conf_mat_vis=plot_conf_mat(stroke_conf_mat).savefig('confusion_matrix.png',dpi=400)


modelHealthScores(strokeModel, training_score, prediction_score)
