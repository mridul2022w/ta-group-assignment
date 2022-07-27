#!/usr/bin/env python
# coding: utf-8
import streamlit as st
import time
import numpy as np
import pandas as pd
import nltk
from bs4 import BeautifulSoup as bs
import re
import os
import codecs
from sklearn import feature_extraction
import autocorrect
from autocorrect import Speller
import requests
import unidecode
import langdetect
from langdetect import detect
import deep_translator
from deep_translator import GoogleTranslator
import pandas as pd


CONTRACTION_MAP = {
"ain't": "is not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't.": "could not.",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he'll've": "he he will have",
"he's": "he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how is",
"i'd": "i would",
"i'd've": "i would have",
"i'll": "i will",
"i'll've": "i will have",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'd've": "it would have",
"it'll": "it will",
"it'll've": "it will have",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she would",
"she'd've": "she would have",
"she'll": "she will",
"she'll've": "she will have",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as",
"that'd": "that would",
"that'd've": "that would have",
"that's": "that is",
"there'd": "there would",
"there'd've": "there would have",
"there's": "there is",
"they'd": "they would",
"they'd've": "they would have",
"they'll": "they will",
"they'll've": "they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what'll've": "what will have",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"when's": "when is",
"when've": "when have",
"where'd": "where did",
"where's": "where is",
"where've": "where have",
"who'll": "who will",
"who'll've": "who will have",
"who's": "who is",
"who've": "who have",
"why's": "why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you would",
"you'd've": "you would have",
"you'll": "you will",
"you'll've": "you will have",
"you're": "you are",
"you've": "you have",
}

#removing new lines and tabs and whitespace
def remove_newlines_tabs(text):
    
    Formatted_text = text.replace('\s+', ' ').replace('. com', '.com')
    
    return Formatted_text

# removing html tags
def rev_html_Tag(text):
    
    soup = bs(text, "html.parser")
    new_text=soup.get_text(separator=" ")
    
    return new_text

#removing any links
def rev_link(text):
    rev_link=re.sub(r'http\S+|\[A-Za-z]*\.com','',text)
    return rev_link

#removing addiotional accented characters from text
def rev_asc(text):
    
    text=unidecode.unidecode(text)
    
    return text
    
#removing puncutations
def rev_puc(text):

    Formatted_text=re.sub(r"[^a-zA-Z0-9:$-,%.?!]+|[()]+",' ',text)
    
    return Formatted_text

#removing repeated charactor
def rev_rep(text):
    
    Pattern_alpha = re.compile(r"([A-Za-z])\1{1,}", re.DOTALL)
    Formatted_text = Pattern_alpha.sub(r"\1\1", text)
    Pattern_Punct = re.compile(r'([.,/#!$%^&*?;:{}=_`~()+-])\1{1,}')
    Combined_Formatted = Pattern_Punct.sub(r'\1', Formatted_text)
    Final_Formatted = re.sub(' {2,}',' ', Combined_Formatted)
    
    return Final_Formatted

#checking the spelling
def spell_check(text):
    
    spell=Speller(lang='en')
    Corrected_text = spell(text)
    
    return Corrected_text

#replacing accented character with alphabat
def replace_accented_character(text):
    
    Formatted_text = text.replace('\x92', "'")
    
    return Formatted_text

    
def call_the_cleaning_func(uber_data):
    
    # # Importing Dataset
    #os.chdir(os.getcwd())
    #uber_data=pd.read_csv("uber_reviews_itune.csv",encoding='latin')


    # proportion of missing values in each variable.
    st.write("Checking for missing values")
    a = uber_data.isna().sum()/uber_data.shape[0]


    # So there is no missing data in this ads

    st.write("Removing Unnecessary columns")
    uber_data_txt = uber_data[['Title','Review']]
    uber_data_txt2=uber_data_txt[['Review']]
    
    st.write("Removing new lines and tabs and whitespace")
    uber_data_txt2['Review_new']=uber_data_txt2['Review'].apply(lambda text:remove_newlines_tabs(text))
    st.write("Removing html tags")
    uber_data_txt2['Review_new2']=uber_data_txt2['Review_new'].apply(lambda text:rev_html_Tag(text))
    st.write("Removing any links")
    uber_data_txt2['Review_new4']=uber_data_txt2['Review_new2'].apply(lambda text:rev_link(text))
    st.write("Replacing accented character with alphabat")
    uber_data_txt2['Review_new5']=uber_data_txt2['Review_new4'].apply(lambda text:replace_accented_character(text))
    st.write("Removing addiotional accented characters from text")
    uber_data_txt2['Review_new6']=uber_data_txt2['Review_new5'].apply(lambda text:rev_asc(text))
    uber_data_txt2['Review_new7']=uber_data_txt2['Review_new6'].str.lower()
    st.write("Removing repeated charactor")
    uber_data_txt2['Review_new8']=uber_data_txt2['Review_new7'].apply(lambda text:rev_rep(text))
    st.write("Removing puncutations")
    uber_data_txt2['Review_new9']=uber_data_txt2['Review_new8'].apply(lambda text:rev_puc(text))

    #expanding the contractions
    st.write("Expanding the contractions")
    Review_new10=[]
    for i in range(0,len(uber_data_txt2)):
        list_of_tokens=uber_data_txt2['Review_new9'].iloc[i].split(' ')
        for Word in list_of_tokens:
            if Word in CONTRACTION_MAP:
                list_of_tokens = [item.replace(Word, CONTRACTION_MAP[Word]) for item in list_of_tokens]
        String_Of_tokens = ' '.join(str(e) for e in list_of_tokens)
        Review_new10.append(String_Of_tokens)
    uber_data_txt2['Review_new10']=Review_new10

    st.write("Detecting the language")
    language=[]
    for i in range(0,len(uber_data_txt2)):
        try:
            k=uber_data_txt2['Review_new10'].iloc[i]
            l=detect(k)
        except Exception as E:
            l='en'
        language.append(l)
            
    uber_data_txt2['language']=language

    st.write("Translating other languages to english")
    Review_new11=[]
    for i in range(0,len(uber_data_txt2)):
        k=uber_data_txt2['Review_new10'].iloc[i]
        n=uber_data_txt2['language'].iloc[i]
        translated = GoogleTranslator(source=n, target='en').translate(k)
        Review_new11.append(translated)
    uber_data_txt2['Review_new11']=Review_new11


    st.write("Expanding the contractions2")
    Review_new12=[]
    for i in range(0,len(uber_data_txt2)):
        list_of_tokens=uber_data_txt2['Review_new11'].iloc[i].split(' ')
        for Word in list_of_tokens:
            if Word in CONTRACTION_MAP:
                list_of_tokens = [item.replace(Word, CONTRACTION_MAP[Word]) for item in list_of_tokens]
        String_Of_tokens = ' '.join(str(e) for e in list_of_tokens)
        Review_new12.append(String_Of_tokens)
    uber_data_txt2['Review_new12']=Review_new12

    st.write("Spell Checker")

    uber_data_txt2['Review_new13']=uber_data_txt2['Review_new12'].apply(lambda text:spell_check(text))
    uber_data_txt2['Review_new14']=uber_data_txt2['Review_new13'].str.lower()

    st.write("Almost Done")
    #removing unnecessary rows and columns
    uber_data_txt3=uber_data_txt2[uber_data_txt2['Review_new13']!='! ! ! ! !']
    uber_data_txt4=uber_data_txt3[['Review_new14']]
    uber_data_txt4=uber_data_txt4.rename(columns={'Review_new14':'Review_new'})
    st.write("Done")
    #exporting final file for further analysis
    uber_data_txt4.to_csv("temp_cleaned.csv")
    st.write(uber_data_txt4)

