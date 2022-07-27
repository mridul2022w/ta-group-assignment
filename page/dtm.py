import numpy as np
import pandas as pd
import nltk
from bs4 import BeautifulSoup
import re
import os
import codecs
from sklearn import feature_extraction
import autocorrect
from autocorrect import Speller
import matplotlib.pyplot as plt
import wordcloud
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from nltk import bigrams
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
import itertools
import collections
import networkx as nx
import streamlit as st
from nltk.stem.snowball import SnowballStemmer

def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens

def generate_basic_wordcloud(data, title):
    cloud = WordCloud(width=400,
                      height=330,
                      max_words=150,
                      colormap='tab20c',
                      stopwords=stopwords,
                      collocations=True).generate_from_text(data)
    plt.figure(figsize=(10,8))
    plt.imshow(cloud)
    plt.axis('off')
    plt.title(title, fontsize=13)
    plt.show()

def dtm_model(uber_review):
    
    uber_review=uber_review.rename(columns={'Review_new':'Review'})

    
    #converting dataframe to list
    uber_review_text = []
    for i in range(0,len(uber_review)):
        text=uber_review['Review'].iloc[i]
        uber_review_text.append(text)

    # Building the DTM
    st.write("Building the DTM")
    vec = CountVectorizer()
    X = vec.fit_transform(uber_review_text)
    dtm = pd.DataFrame(X.toarray(), columns=vec.get_feature_names())
    rowNames=['doc ' + format(x+1, '01d') for x in range(dtm.shape[0])]
    rowNames_series = pd.Series(rowNames)
    dtm.rename(index=rowNames_series, inplace=True)
    st.write(dtm)
    
    st.write("Building the DTM for TFIDF")
    # Building the DTM for TFIDF weighing
    tfidf_vectorizer = TfidfVectorizer()

    tfidf_matrix=tfidf_vectorizer.fit_transform(uber_review_text)
    tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=vec.get_feature_names())
    rowNames=['doc ' + format(x+1, '01d') for x in range(tfidf.shape[0])]
    rowNames_series = pd.Series(rowNames)
    tfidf.rename(index=rowNames_series, inplace=True)
    st.write(tfidf)

    # # creating bigrams
    st.write("Creating bigrams")
    words_in_tweet = [tweet.lower().split() for tweet in uber_review_text]
    terms_bigram = [list(bigrams(tweet)) for tweet in words_in_tweet]

    bigram = list(itertools.chain(*terms_bigram))

    # Create counter of words in clean bigrams
    bigram_counts = collections.Counter(bigram)
    bigram_df = pd.DataFrame(bigram_counts.most_common(50),
                                 columns=['bigram', 'count'])
    st.dataframe(bigram_df)

    # # COG plot
    # Create dictionary of bigrams and their counts
    d = bigram_df.set_index('bigram').T.to_dict('records')

    # Create network plot 
    G = nx.Graph()

    # Create connections between nodes
    for k, v in d[0].items():
        G.add_edge(k[0], k[1], weight=(v * 10))

    fig, ax = plt.subplots(figsize=(10, 8))

    pos = nx.spring_layout(G, k=2)

    # Plot networks
    st.write("COG Plot")
    nx.draw_networkx(G, pos,
                     font_size=16,
                     width=3,
                     edge_color='grey',
                     node_color='purple',
                     with_labels = False,
                     ax=ax)

    # Create offset labels
    for key, value in pos.items():
        x, y = value[0]+.135, value[1]+.045
        ax.text(x, y,
                s=key,
                bbox=dict(facecolor='red', alpha=0.25),
                horizontalalignment='center', fontsize=13)
        
    plt.show()
    st.pyplot(plt)
    