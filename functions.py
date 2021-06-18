import joblib
import numpy as np
from flask import request


# takes form data and formats it for classifer model
def formatPtDataalt(req):
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
    ptData[7] = req.get('age')
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


def formatPtData(req):
    ptData = [0, 0., 0., 0., 0, 70., 0., 0., 190., 0, 0, 0, 0., 0,
              0, 0., 0.]

    sx = ['facial_deficit', 'ARM_DEFICIT', 'LEG_DEFICIT', 'DYSPHASIA', 'Visuospatial_disorder', 'Hemianopia',
          'Brainstem_cerebellar_signs', 'Other_deficit']

    # dataKEY
    # 0 = Drowsy
    # 1 = Alert
    # 2 = unresponsive
    if req.get('ms') == 'alert':
        ptData[0] = 0
        ptData[1] = 1
    else:
        ptData[0] = 1.0
        ptData[1] = 0.0
    # 3 = female
    # 4 = male
    if req.get('gender') == 'male':
        ptData[3] = 0.0
        ptData[4] = 1.0
    else:
        ptData[4] = 1.0
        ptData[3] = 0.0
    # 5 = age
    ptData[5] = req.get('age')

    # 6 = rsleep
    if req.get('wake_up_cva') == 'wake_up_cva':
        ptData[6] = 1
    # 7 = rAtrial
    if req.get('afib') == 'on':
        ptData[7] = 1
    # 8= sysBP
    ptData[8] = req.get('bp')

    for s in sx:
        i = sx.index(s) + 9
        print(s,i)

        print(req.get(s))
        if req.get(s) == s:
            ptData[i] = 1
        else:
            ptData[i] = 0

    return ptData








def load_model(ptData):
    strokeModel = joblib.load('sx_clf_rf.pkl')
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


# update Database with patient data after dx has been confirmed

def updateDB():
    if request.method == "POST":
        req = request.form
        print(req)
