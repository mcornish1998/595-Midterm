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

def pol_sub(text):
    try:
        blob = TextBlob(text)
        polarity = blob.sentiment[0]
        subjectivity = blob.sentiment[1]
        return polarity, subjectivity
    except:
        return 'Error calculating subjectivity'


def commonwords(text):
    try:
        words = TextBlob(text).words
        wordFrame = pd.DataFrame(words, columns=['Words'])
        mostCommonlyUsed = wordFrame['Words'].value_counts()[:3]
        return mostCommonlyUsed.to_dict()
    except:
        return 'Error finding common words'

def stop_filter(text):
    stop_words = set(stopwords.words("english"))
    tok = word_tokenize(text)
    filtered_sent = []
    for w in tok:
        if w not in stop_words:
            filtered_sent.append(w)
    return filtered_sent

def pos_count(text):
    try:
        words = nltk.word_tokenize(text)
        tags = nltk.pos_tag(words)
        pos_list = []
        for x in tags:
            pos_list.append(x[1])
        pos_list = pd.Series(pos_list)
        pos_list = pos_list.value_counts()
        return pos_list.to_dict()

    except Exception as e:
        print(e)
        return 'Error counting parts of speech'
