import pandas as pd
from flask import Flask, render_template

from functions import runModel

app = Flask(__name__)


@app.route('/dashboard')
def home():
    return render_template("index.html")


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
    return render_template("dashboard.html", strokeProb="P(CVA)")


@app.route("/new_patient", methods=["POST", "GET"])
def formSubmit():
    prob = runModel()
    return render_template("dashboard.html", strokeProb=prob)


if __name__ == "__main__":
    app.run()
