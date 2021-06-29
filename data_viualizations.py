# data_vizualizations.py
# this file contains methods that display data from logic formatted for the Front end
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from bokeh.io import export_png
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

from bokeh.plotting import figure
from bokeh.resources import CDN
from bokeh.embed import file_html
import numpy as np
from bokeh.io import show, output_file
from bokeh.plotting import figure


# All DATA PAGE


# AGE & CVA
def graphAgeCVA(data):
    hist, edges = np.histogram(data.AGE, density=True, bins=20)
    agePlot = figure(title='CVA & AGE')
    agePlot.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], line_color="white")
    agePlot.xaxis.axis_label = 'AGE'
    agePlot.yaxis.axis_label = 'n patients'
    export_png(agePlot, filename="static/graphAgeCva.png")


# Vessel INVOLVED BY SX

def vesselInvolved(data):
    i = data[0][0]
    h = data[0][1]
    v = figure(title="I/H", x_axis_label='P(I) vs. P(H)')
    v.scatter(i, 0, size=i * 100, color='blue', alpha=.5)
    v.scatter(h, 0, size=h * 100, color='red', alpha=.5)
    export_png(v, filename="static/vesselInvolved.png")


def BpAGE(data):
    BpPlot = figure(title='BP and AGE in CVA')
    BpPlot.scatter(data.RSBP, data.AGE, color='red', alpha=0.1, size=1)
    BpPlot.xaxis.axis_label = 'BP'
    BpPlot.yaxis.axis_label = 'AGE'
    export_png(BpPlot, filename="static/BpPlot.png")


def create_vis_ih(strokeType):
    i = strokeType[0][0]
    h = strokeType[0][1]
    v = figure(title="I/H", x_axis_label='P(I) vs. P(H)')
    v.scatter(i, 0, size=i * 100, color='blue', alpha=.5)
    v.scatter(h, 0, size=h * 100, color='red', alpha=.5)
    export_png(v, filename="static/ihprob.png")
    return True

#  DATA HEALTH PAGE


#  Stroke Diagnostics
