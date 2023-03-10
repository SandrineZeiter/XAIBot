# ----------------------- imports -----------------------
# imports for flask
import base64
import os

from flask import request, Flask, jsonify

# imports for ML
import random

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# imports for LIME
from lime.lime_text import LimeTextExplainer

# import matplotlib
import matplotlib


# ----------------------- Definition of the plot -----------------------
# To save the figure
matplotlib.use("Agg")
matplotlib.rc('xtick', labelsize=14)
matplotlib.rc('ytick', labelsize=14)
matplotlib.rc('axes', titlesize=16)
matplotlib.rc('axes', labelsize=16)
matplotlib.rc('figure', titlesize=20)

# ----------------------- ML-part -----------------------
# import dataset
data = pd.read_csv("/Users/sandrinezeiter/Library/CloudStorage/OneDrive-UniversitédeFribourg/"
                   "Thesis/Twitter_cleaned.csv")

# define the different class names for feelings
class_names = ['afraid', 'alive', 'angry', 'confused', 'depressed', 'good', 'happy',
               'helpless', 'hurt', 'indifferent', 'interested', 'love', 'open', 'positive',
               'sad', 'strong']

# split into training and test set
train, test = train_test_split(data, train_size=0.9)

# tokenize the sentences
vectorizer = TfidfVectorizer(analyzer='word', token_pattern=r'\b[a-zA-Z]{3,}\b', lowercase=False,
                             min_df=5, max_df=0.7, stop_words='english')
vectorizer.fit_transform(train['tweet'])

rfc = RandomForestClassifier()
mnb = MultinomialNB(alpha=0.1)
p1 = make_pipeline(vectorizer, mnb)
# p1 = make_pipeline(vectorizer, rfc)

alpha_grid = np.logspace(-3, 0, 4)  # Is smoothing parameter for the counts
param_grid = [{'multinomialnb__alpha': alpha_grid}]
gs = GridSearchCV(p1, param_grid=param_grid, cv=5, return_train_score=True)

gs.fit(train.tweet, train.feeling)