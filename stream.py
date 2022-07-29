import streamlit as st
import pandas as pd
import numpy as np
import os
from page.Cleaning import *
from page.sa import *
from page.token import *
from page.dtm import *
from page.ltm import *
from page.metadata import *
import random

def save_uploadedfile(uploadedfile):
    st.session_state['key'] = random.randint(0,99999)
    with open(os.path.join("temp"+str(st.session_state['key'])+".csv"),"wb") as f:
        f.write(uploadedfile.getbuffer())
    
    return st.success("Saved File:{}".format(uploadedfile.name))

def main_page():
    st.markdown("## Data Upload")
    func_check()
    
    # if st.button("Next"):
    #     st.session_state.runpage = page7()


def page2():
    st.markdown("## Data Cleaning")
    if "key" in st.session_state:
        if os.path.isfile("temp"+str(st.session_state['key'])+".csv"):
            if os.path.isfile("temp_cleaned"+str(st.session_state['key'])+".csv"):
                data = pd.read_csv("temp_cleaned"+str(st.session_state['key'])+".csv",encoding='cp1252')
                st.write(data)
            else:
                data = pd.read_csv("temp"+str(st.session_state['key'])+".csv",encoding='cp1252')
                call_the_cleaning_func(data)
        else:
            st.write("Please upload data to proceed further")
    else:
        st.write("Please upload data to proceed further")

def page3():
    st.markdown("## Sentimental Analysis")
    if "key" in st.session_state:
        if os.path.isfile("temp"+str(st.session_state['key'])+".csv"):
            if os.path.isfile("temp_cleaned"+str(st.session_state['key'])+".csv"):
                data = pd.read_csv("temp_cleaned"+str(st.session_state['key'])+".csv",encoding='cp1252')
                call_sa(data)
            else:
                st.write("Please clean data to proceed further")
        else:
            st.write("Please upload data to proceed further")
    else:
        st.write("Please upload data to proceed further")
    
def page4():
    st.markdown("## Tokenization and Stemming")
    if "key" in st.session_state:
        if os.path.isfile("temp"+str(st.session_state['key'])+".csv"):
            if os.path.isfile("temp_cleaned"+str(st.session_state['key'])+".csv"):
                data = pd.read_csv("temp_cleaned"+str(st.session_state['key'])+".csv",encoding='cp1252')
                call_tokenization(data)
            else:
                st.write("Please clean data to proceed further")
        else:
            st.write("Please upload data to proceed further")
    else:
        st.write("Please upload data to proceed further")
    

def page5():
    st.markdown("## DTM & IDF")
    if "key" in st.session_state:
        if os.path.isfile("temp"+str(st.session_state['key'])+".csv"):
            if os.path.isfile("temp_cleaned"+str(st.session_state['key'])+".csv"):
                data = pd.read_csv("temp_cleaned"+str(st.session_state['key'])+".csv",encoding='cp1252')
                dtm_model(data)
            else:
                st.write("Please clean data to proceed further")
        else:
            st.write("Please upload data to proceed further")
    else:
        st.write("Please upload data to proceed further")

def page6():
    st.markdown("## LTM Model")
    if "key" in st.session_state:
        if os.path.isfile("temp"+str(st.session_state['key'])+".csv"):
            if os.path.isfile("temp_cleaned"+str(st.session_state['key'])+".csv"):
                data = pd.read_csv("temp_cleaned"+str(st.session_state['key'])+".csv",encoding='cp1252')
                call_ltm(data)
            else:
                st.write("Please clean data to proceed further")
        else:
            st.write("Please upload data to proceed further")
    else:
        st.write("Please upload data to proceed further")

def page7():
    st.markdown("## Change Metadata")
    if "key" in st.session_state:
        if os.path.isfile("temp"+str(st.session_state['key'])+".csv"):
            data = pd.read_csv("temp"+str(st.session_state['key'])+".csv",encoding='cp1252')
            meta_data(data)
        else:
            st.write("Please upload data to proceed further")
    else:
        st.write("Please upload data to proceed further")

page_names_to_funcs = {
    "Upload": main_page,
    "Metadata":page7,
    "Clean": page2,
    "Sentimental Analysis": page3,
    "Tokenization & Stemming": page4,
    "DTM & IDF Model": page5,
    "LTM Model":page6
}

def func_check():
    
    
    st.write('')
    with open('uber_reviews_itune.csv', 'rb') as f:
        st.download_button(
         label="Download data for this app",
         data = f,
         file_name='uber_reviews_itune.csv',
         mime='text/csv')
    
    uploaded_file = st.file_uploader("Choose a file",type=["csv"])
    
    if uploaded_file is not None:
         # To read file as bytes:
         try:
             data = pd.read_csv(uploaded_file,encoding='cp1252')
         except Exception as e:
             st.write("Error",e)
         finally:
             save_uploadedfile(uploaded_file)
             st.dataframe(data)
    else:
        if "key" in st.session_state:
            if os.path.isfile("temp"+str(st.session_state['key'])+".csv"):
                os.remove("temp"+str(st.session_state['key'])+".csv")
            if os.path.isfile("temp_cleaned"+str(st.session_state['key'])+".csv"):
                os.remove("temp_cleaned"+str(st.session_state['key'])+".csv")
    
    
  
def main():
    selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
    with st.spinner('Wait for it...'):
        time.sleep(0.5)
        page_names_to_funcs[selected_page]()
        
if __name__ == '__main__':
    main()