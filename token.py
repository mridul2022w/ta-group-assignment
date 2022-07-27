import time
import numpy as np
import pandas as pd
import nltk
import re
import os
import codecs
from sklearn import feature_extraction
import autocorrect
from autocorrect import Speller
import matplotlib.pyplot as plt
import streamlit as st
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import wordcloud
from nltk.stem.snowball import SnowballStemmer
nltk.download('punkt')


def tokenize_and_stem(text):
    
    stemmer = SnowballStemmer("english")
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation) using regex
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems

def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens

def call_tokenization(uber_review):
    
    uber_review=uber_review.drop(['Unnamed: 0'],axis=1)
    uber_review_text = uber_review['Review_new'].tolist()
    output3=[''.join(uber_review_text)]
    
    
    #tokenization
    st.write("Tokenization")
    token=tokenize_only(str(output3))
    #removing stopwords
    st.write("Stopword removal")
    # load nltk's English stopwords as variable called 'stopwords'
    stopwords = nltk.corpus.stopwords.words('english')
    stopwords.extend(['a', 'an', 'the', 'to', 'for','also','wold'])
    tokens_without_sw = [word for word in token if not word in stopwords]
    
    #creating a datafarme
    tokens_without_sw_df=pd.DataFrame(tokens_without_sw,columns=['Words'])
    #grouping works and sorting
    tokens_without_sw_grp=tokens_without_sw_df.groupby('Words').agg({'Words':[np.size]})
    tokens_without_sw_grp.columns=[''.join(col) for col in tokens_without_sw_grp.columns]
    tokens_without_sw_grp2=tokens_without_sw_grp.sort_values('Wordssize',ascending=False)
    tokens_without_sw_grp2=tokens_without_sw_grp2.reset_index()
    #keeping top words
    st.write("Frequency of top 20 words without stemming")
    tokens_without_sw_grp3=tokens_without_sw_grp2.head(20)
    tokens_without_sw_grp4=tokens_without_sw_grp3.sort_values('Wordssize',ascending=True)
    #creating bar chart to determine top words from list
    tokens_without_sw_grp4.plot(x='Words',y='Wordssize',kind='barh')
    plt.title("Top word frequency")  # add title
    plt.xlabel("Words")
    plt.ylabel("Count")
    plt.grid()
    plt.show()
    st.pyplot(plt)
    
    st.write("Stemming")
    stem=tokenize_and_stem(str(output3))
    stem2 = [word for word in stem if not word in stopwords]

    
    st.write("Wordcloud")
    #creating worldcould
    text = ' '.join(stem2).lower()
    wordcloud = WordCloud(stopwords = STOPWORDS,collocations=True).generate(str(text))
    #plot the wordcloud object
    plt.figure(figsize=(20,10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title("")
    plt.tight_layout()
    plt.axis('off')
    plt.show()
    st.pyplot(plt)
    
    #creating frequncy of words
    text_dictionary = wordcloud.process_text(str(text))
    word_freq={k: v for k, v in sorted(text_dictionary.items(),reverse=True, key=lambda item: item[1])}
    rel_freq=wordcloud.words_

    #creating a datafarme
    stem_df=pd.DataFrame(word_freq.items(),columns=['Word', 'Frequency'])
    stem_df2=stem_df.head(20)
    st.write("Frequency of top 20 words")
    #creating bar chart to determine top words from list
    stem_df2.plot(x='Word',y='Frequency',kind='bar',figsize=(10,10))
    plt.title("Top word frequency")  # add title
    plt.xlabel("Words")
    plt.ylabel("Count")
    plt.grid()
    plt.show()
    st.pyplot(plt)

