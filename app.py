import pandas as pd
from flask import Flask, render_template, request, url_for, redirect

from functions import runModel

app = Flask(__name__)

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
    stroke_data = pd.read_csv('data/stroke_data.csv')
    return render_template("all-data.html", tables=[stroke_data.to_html(classes='stroke_data')],
                           titles=stroke_data.columns.values)


@app.route('/datahealth')
def datahealth():
    accu = 74
    clr = pd.read_pickle("data/classification_report.pkl")
    return render_template("datahealth.html", accu=accu, tables=[clr.to_html(classes='clr')], titles=clr.columns.values)


@app.route('/dashboard')
def dashboard():
    return render_template("dashboard.html", len=len(diseases), diseases=diseases, strokeProb="Enter Pt. Data to see P(CVA)")


@app.route("/new_patient", methods=["POST", "GET"])
def formSubmit():
    prob = runModel()
    return render_template("dashboard.html", strokeProb=prob, len=len(diseases), diseases=diseases)


if __name__ == "__main__":
    app.run()
