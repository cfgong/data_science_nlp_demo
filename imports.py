import nltk
from nltk.tokenize import casual_tokenize
nltk.download('averaged_perceptron_tagger')

from nltk.corpus import stopwords
stopWords = set(stopwords.words('english'))

import pandas as pd
import numpy as np

import psycopg2
import spacy
from spacy.matcher import PhraseMatcher
nlp = spacy.load("en_core_web_lg")

import re

import warnings
warnings.filterwarnings('ignore')

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix

print("package imports done")
