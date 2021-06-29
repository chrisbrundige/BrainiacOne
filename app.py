import pandas as pd
from flask import Flask, render_template, request, url_for, redirect
from data_visualizations import BpAGE, graphAgeCVA
from functions import runModel, predictType, updateDB
from data_visualizations import create_vis_ih

app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'
prob = '(P|CVA)'
diseases = ['Brain Tumors',
            'Hypoglycemia',
            'Toxic poisoning',
            'Hysterical attacks',
            'Seizure',
            'Sepsis',
            'Subdural hematoma',
            'Uremia',
            'Vestibulopathy',
            'Multiple sclerosis',
            'Brain Metastasis',
            'Encephalopathy',
            'Hyperglycemia ',
            'Syncope',
            'Hyponatremia',
            'Hyperkalemia',
            'Spinal cord Lesions',
            'Hypothyroidism',
            'Encephalitis',
            'Dementia',
            'Migraine']
sx = ['facial_deficit', 'ARM_DEFICIT', 'LEG_DEFICIT', 'DYSPHASIA', 'Visuospatial_disorder', 'Hemianopia',
      'Brainstem_cerebellar_signs', 'Other_deficit']


# Route for handling the login page logic


@app.route('/', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        if request.form['username'] != 'admin' or request.form['password'] != 'admin':
            error = 'Invalid Credentials. Please try again.'
        else:
            return redirect(url_for('dashboard'))
    return render_template('login.html', error=error)


@app.route('/all-data')
def alldata():
    stroke_data = pd.read_csv('data/stroke_data_complete.csv')
    BpAGE(stroke_data)
    graphAgeCVA(stroke_data)
    stroke_data = stroke_data[1000:1500]
    return render_template("all-data.html", tables=[stroke_data.to_html(classes='stroke_data')],
                           titles=stroke_data.columns.values)


@app.route('/datahealth')
def datahealth():
    clr = pd.read_pickle("data/classification_report.pkl")
    return render_template("datahealth.html", tables=[clr.to_html(classes='clr')], titles=clr.columns.values)


@app.route('/dashboard')
def dashboard():
    return render_template("dashboard.html", len=len(diseases), lensx=len(sx), diseases=diseases, sx=sx,
                           strokeProb="Enter Pt. Data to see P(CVA)")


@app.route("/new_patient", methods=["POST", "GET"])
def formSubmit():
    try:
        create_vis_ih(predictType())
        prob = runModel()
        return render_template("dashboard.html", lensx=len(sx), strokeProb=prob, len=len(diseases), diseases=diseases,
                               sx=sx)
    except():
        print('err')
        return render_template("generic-error.html", err_msg="Some Required values are missing", err="NULL VALUE ERROR")


@app.route("/confirm_dx", methods=["POST", "GET"])
def confirmDX():
    updateDB()
    return render_template("dashboard.html", strokeProb=prob, len=len(diseases), diseases=diseases, lensx=len(sx),
                           sx=sx)


if __name__ == "__main__":
    app.run()
