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
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.parsing.preprocessing import strip_punctuation, strip_tags, strip_numeric
from nltk.stem.wordnet import WordNetLemmatizer   
import string
from pprint import pprint
import matplotlib.pyplot as plt


def textClean(text0):
    #text1 = [strip_punctuation(doc) for doc in text0]
    #text1 = [strip_tags(doc) for doc in text1]
    #text1 = [strip_numeric(doc) for doc in text1]
    text1 = [[word for word in ''.join(doc).split()] for doc in text0]
    normalized = [[" ".join([word for word in ' '.join(doc).split()])] for doc in text1]
    return normalized

def build_beta_df(lda_model, id2word):
    beta = lda_model.get_topics()  # shape (num_topics, vocabulary_size).
    beta_df = pd.DataFrame(data=beta)

    # convert colnames in beta_df 2 tokens
    token2col = list(id2word.token2id)
    beta_df.columns = token2col
    # beta_df.loc[0,:].sum()  # checking if rows sum to 1

    # convert rownames too, eh? Using format(), .shape[] and range()
    rowNames=['topic' + format(x+1, '02d') for x in range(beta_df.shape[0])]
    rowNames_series = pd.Series(rowNames)
    beta_df.rename(index=rowNames_series, inplace=True)
    return(beta_df)


def compute_perplexity_values(model_list, corpus, start, limit, step):
    perplexity_values = []
    for num_topics in range(start, limit, step):
        #model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=num_topics, random_state=100,
        #                                  update_every=1, chunksize=100, passes=10, alpha='auto', per_word_topics=True)
        model_index = num_topics - start
        model = model_list[model_index]
        perplexity_values.append(model.log_perplexity(corpus))
        #model_list.append(model)
        

    return perplexity_values  # note, list of 2 objs returned


## compute coherence score (akin to LMD?)
def compute_coherence_values(dictionary, corpus, texts, limit, start, step):
    coherence_values = []
    model_list = []
    num_topics1 = [i for i in range(start, limit, step)]
    for num_topics in num_topics1:
        model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=num_topics, random_state=100,
                                           update_every=1, chunksize=100, passes=10, alpha='auto', per_word_topics=True)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values  # note, list of 2 objs returned

def build_gamma_df(lda_model, corpus0):
    gamma_doc = []  # empty list 2 populate with gamma colms
    num_topics = lda_model.get_topics().shape[0]
    
    for doc in range(len(corpus0)):
        doc1 = corpus0[doc].split()
        bow_doc = id2word.doc2bow(doc1)
        gamma_doc0 = [0]*num_topics  # define list of zeroes num_topics long
        gamma_doc1 = lda_model.get_document_topics(bow_doc)
        gamma_doc2_x = [x for (x,y) in gamma_doc1]#; gamma_doc2_x
        gamma_doc2_y = [y for (x,y) in gamma_doc1]#; gamma_doc2_y
        for i in range(len(gamma_doc1)):
            x = gamma_doc2_x[i]
            y = gamma_doc2_y[i]
            gamma_doc0[x] = y  # wasn't geting this in list comprehension somehow 
        gamma_doc.append(gamma_doc0)
        
    gamma_df = pd.DataFrame(data=gamma_doc)  # shape=num_docs x num_topics
    topicNames=['topic' + format(x+1, '02d') for x in range(num_topics)]
    topicNames_series = pd.Series(topicNames)
    gamma_df.rename(columns=topicNames_series, inplace=True)
    return(gamma_df)


def call_ltm_model(uber_review):
    
    #keeping only required coulmn
    uber_review=uber_review[['Review_new3']]
    uber_review=uber_review.rename(columns={'Review_new3':'Review'})


    #converting dataframe to list
    uber_review_text = []
    for i in range(0,len(uber_review)):
        text=uber_review['Review'].iloc[i]
        uber_review_text.append(text)


     # for the .join() func

    lemma = WordNetLemmatizer()
    corpus1 = textClean(uber_review_text)
    corpus2 = [[word for word in ' '.join(doc).split()] for doc in corpus1]

    id2word = corpora.Dictionary(corpus2)  # Create Dictionary
    corpus = [id2word.doc2bow(text) for text in corpus2]
    a0 = [[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]]

    # Build LDA model for (say) K=4 topics
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                               id2word=id2word,
                                               num_topics=4, 
                                               random_state=100,
                                               update_every=1,
                                               chunksize=100,
                                               passes=10,
                                               alpha='auto',
                                               per_word_topics=True)

    ## obtain the factor matrices - beta
    print(lda_model)
    # invoke func
    beta_df = build_beta_df(lda_model=lda_model, id2word=id2word)
    beta_df

    # func to get gamma matrix by looping using list.comp

    # now apply func
    gamma_df = build_gamma_df(lda_model=lda_model, corpus0=uber_review_text)
    gamma_df

    row0 = gamma_df.values.tolist()
    row=[]
    for i in range(len(row0)):
        row1 = list(enumerate(row0[i]))
        row1_y = [y for (x,y) in row1]
        max_propn = sorted(row1_y, reverse=True)[0]
        row2 = [(i, x, y) for (x, y) in row1 if y==max_propn]
        row.append(row2)

    row

    # Can take a long time to run.
    start1=2
    limit1=19
    step1=1

    model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, 
                                                            texts=corpus2, start=start1, limit=limit1, step=step1)

    print(coherence_values)


    # In[30]:


    # obtain optimal topic number
    coher = list(enumerate(coherence_values))  # create an index for each list elem
    index_max = [x for (x,y) in coher if y==max(coherence_values)]  # obtain index num corres to max coherence value
    Optimal_numTopics = int(str(index_max[0]))+2  # convert that list elem into integer (int()) via string (str())
    print(Optimal_numTopics)    


    # In[31]:


    ## Plot the change in coherence score with num_topics
    get_ipython().run_line_magic('matplotlib', 'inline')

    start1=2
    limit1=19
    step1=1
    # Show graph
    x = range(start1, limit1, step1)
    plt.plot(x, coherence_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.axvline(x=Optimal_numTopics, color='r')
    plt.show()


    ## compute perplexity fit

    # Can take a long time to run.
    # perplexity_values = compute_perplexity_values(dictionary=id2word, corpus=corpus, start=2, limit=15, step=1)
    perplexity_values = compute_perplexity_values(model_list, corpus=corpus, start=start1, limit=limit1, step=step1)
    print(perplexity_values)



    # compute optimal num_topics using perplexity based fit
    perpl = list(enumerate(perplexity_values))  # create an index for each list elem
    index_min = [x for (x,y) in perpl if y==min(perplexity_values)]  # obtain index num corres to max coherence value
    optimal_numTopics = int(str(index_min[0]))+2  # convert that list elem into integer (int()) via string (str())
    print(optimal_numTopics)  



    # graph the perplexity fit and see
    get_ipython().run_line_magic('matplotlib', 'inline')

    # Show graph
    x = range(start1, limit1, step1)
    plt.plot(x, perplexity_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Perplexity score")
    plt.legend(("perplexity_values"), loc='best')
    plt.axvline(x=optimal_numTopics, color='r')
    plt.show()



    # seems optimal num_topics is 10
    optimal_model = model_list[Optimal_numTopics]
    model_topics = optimal_model.show_topics(formatted=False)
    pprint(optimal_model.print_topics(num_words=10))


    # Get main topic in each document
    gamma_df = build_gamma_df(lda_model=optimal_model, corpus0=uber_review_text)
    #gamma_df.iloc[:8,:8]

    row0 = gamma_df.values.tolist()
    row=[]
    for i in range(len(row0)):
        row1 = list(enumerate(row0[i]))
        row1_y = [y for (x,y) in row1]
        max_propn = sorted(row1_y, reverse=True)[0]
        row2 = [(i, x, y) for (x, y) in row1 if y==max_propn]
        row.append(row2)

    row

    sent_topics_df = pd.DataFrame()
    for row1 in row:
        for (doc_num, topic_num, prop_topic) in row1:
            wp = optimal_model.show_topic(topic_num)
            topic_keywords = ", ".join([word for word, prop in wp])
            sent_topics_df = sent_topics_df.append(pd.Series([int(doc_num), int(topic_num), 
                                                              round(prop_topic,4), 
                                                              topic_keywords]), 
                                                           ignore_index=True)
        
    sent_topics_df.columns = ['Doc_num', 'Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']    
    sent_topics_df



    # Add original text to the end of the output
    contents = pd.Series(uber_review_text)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    #return(sent_topics_df)
    sent_topics_df.columns = ['Doc_num', 'Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords', 'contents']
    sent_topics_df
