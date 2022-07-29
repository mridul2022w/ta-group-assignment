import requests, time, re
import pandas as pd
from bs4 import BeautifulSoup
import vaderSentiment
import nltk
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk import sent_tokenize, word_tokenize
import matplotlib.pyplot as plt
import os
import streamlit as st
nltk.download('punkt')


#removing all $ signs
def rev_dollar(text):
    
    rev_dollar=re.sub(r'\$','',text)
    
    return rev_dollar

#function to call all docs
def vader_wrap_func(analyzer,corpus0):
    
    # define empty DF to concat unit func output to
    vs_df = pd.DataFrame(columns=['doc_index', 'sent_index', 'neg', 'neu', 'pos', 'compound', 'sentence'])    
    
    # apply unit-func to each doc & loop over all docs
    for i1 in range(len(corpus0)):
        doc0 = (corpus0['Review'].iloc[i1])
        vs_doc_df = vader_unit_func(analyzer,doc0)  # applying unit-func
        vs_doc_df.insert(0, 'doc_index', i1)  # inserting doc index
        vs_df = pd.concat([vs_df, vs_doc_df], axis=0)
        
    return(vs_df)


# defining unit func to process one doc
def vader_unit_func(analyzer,doc0):
    
    sents_list0 = sent_tokenize(doc0)
    vs_doc0 = []
    sent_ind = []
    for i in range(len(sents_list0)):
        vs_sent0 = analyzer.polarity_scores(sents_list0[i])
        vs_doc0.append(vs_sent0)
        sent_ind.append(i)
        
    # obtain output as DF    
    doc0_df = pd.DataFrame(vs_doc0)
    doc0_df.insert(0, 'sent_index', sent_ind)  # insert sent index
    doc0_df.insert(doc0_df.shape[1], 'sentence', sents_list0)
    
    return doc0_df

def call_sa(uber_review):
    
    st.write('Analyzing the dataset')
    uber_review=uber_review.drop(['Unnamed: 0'],axis=1)
    uber_review['Review_new2']=uber_review['Review_new'].apply(lambda text:rev_dollar(text))
    uber_review=uber_review.drop(['Review_new'],axis=1)
    uber_review=uber_review.rename(columns={'Review_new2':'Review'})
    
    analyzer = SentimentIntensityAnalyzer()
    
    #defining the sentiment ADS
    st.write('Computing the sentiment score')
    uber_review_vs=vader_wrap_func(analyzer,uber_review)
    uber_review_vs2=uber_review_vs.drop(['sentence'],axis=1)
    review_score=uber_review_vs2.groupby('doc_index').sum()
    st.write('Summary of the sentiment score')
    review_score

    #re-arrangeing ADS for charting
    uber_review_vs2=uber_review_vs.sort_values(by="compound",ascending=True)
    sen=[]
    comp_score=[]
    for i in range(0,len(uber_review_vs2)):
        comp_score2=uber_review_vs2['compound'].iloc[i]
        comp_score.append(comp_score2)
        sen.append(i)
        
    doc0_df=pd.DataFrame()
    doc0_df.insert(0,'Sentence number',sen)
    doc0_df.insert(1,'Compound valence score',comp_score)
    
    # line-graph of sentimt scores via matplotlib 
    #charting
    st.write('Valence of Uber Reviews by sentence')
    doc0_df.plot(x='Sentence number',y='Compound valence score',kind='bar')
    plt.title("Valence of Uber Reviews by sentence")  # add title
    plt.ylabel("Compound valence score")
    plt.xticks(np.arange(0, 1840, 250),rotation=0)
    st.pyplot(plt)

    doc0_df['emotion'] = np.where(doc0_df['Compound valence score'] > 0,'positive',np.where(doc0_df['Compound valence score'] < 0,'negative','neutral'))
    
    emotion_share=doc0_df.groupby('emotion').agg({'Sentence number':[np.size]})
    emotion_share.columns=[''.join(col) for col in emotion_share.columns]
    emotion_share=emotion_share.rename(columns={'Sentence numbersize':'size'}).reset_index()
    emotion_share['percent']=(emotion_share['size']/(emotion_share['size'].sum()))
    emotion_share=emotion_share.drop(['size'],axis=1)
    emotion_share.set_index('emotion',inplace=True)
    
    st.write('% Share of score')
    st.write(emotion_share)
    
    st.write('Done')