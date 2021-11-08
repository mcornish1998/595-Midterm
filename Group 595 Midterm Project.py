import flask
import nltk
from textblob import TextBlob
from textblob import Word
import numpy as np
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('averaged_perceptron_tagger')




def sentiment(text):
    try:
        results = []
        analyzer = SentimentIntensityAnalyzer()
        scores = analyzer.polarity_scores(text)
        results.append([scores])
        return results
    except:
        return 'Error calculating sentiment'


def remove_apostrophes(text):
    try:
        text = text.replace("'", "")
        text = text.replace('"', "")
        text = text.replace('`', "")
        return text
    except:
        return 'Error removing apostrophes'
