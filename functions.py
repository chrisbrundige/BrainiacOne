import joblib
import numpy as np
from flask import request


# takes form data and formats it for classifer model
def formatPtData(req):
    # test data key
    ptData = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 49.0, 0.0, 0.0, 171.23, 34.400000]
    # 0 =  female
    # 1 = male
    # 2 = other gender
    if req.get('gender') == 'male':
        ptData[0] = 0.0
        ptData[1] = 1.0
    else:
        ptData[0] = 1.0
        ptData[1] = 0.0
    # 3 = smoker unknown
    # 4 = former smoker
    # 5 = never smoked
    # 6 = current smoker
    if req.get('smokes') == 'on':
        ptData[6] = 1.0
    elif req.get('never smoked') == 'on':
        ptData[5] = 1.0
    elif req.get('formerly smoked') == 'on':
        ptData[4] = 1.0
    else:
        ptData[3] = 1.0
    # 7 =age
    ptData[7]=req.get('age')
    # 8 = hx of HTN
    if req.get('Hypertension') == 'on':
        ptData[8] = 1.0
    # 9 = hx of heart disease
    if req.get('Heart Disease') == 'on':
        ptData[9] = 1.0
    # 10 = bgl
    ptData[10] = req.get('bgl')
    # 11 bmi
    ptData[11] = req.get('bmi')
    print(ptData)
    print(req)
    return ptData


def load_model(ptData):
    strokeModel = joblib.load('strokePred.pkl')
    ptData = np.array(ptData)
    # reshape data to handle single patient data
    ptData = ptData.reshape(1, -1)
    isCva = strokeModel.predict(ptData)
    prob = strokeModel.predict_proba(ptData)
    probmsg = f' probability of Stroke is {prob[0][1] * 100} % prediction value {isCva}'
    print(probmsg)
    return probmsg

# gets new patient form data
## this data has NOT been formatted for ML model
def newPatient():
    if request.method == "POST":
        req = request.form
        # print(req)
        return req
    # return redirect(request.url)


def runModel():
    prob = load_model(formatPtData(newPatient()))
    return prob


# return model score
def modelHealth():
    print("everything's fine")
