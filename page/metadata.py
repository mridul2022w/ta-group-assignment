import pandas as pd
import streamlit as st 
import os

def meta_data(data):
    """This application is created to help the user change the metadata for the uploaded file. 
    They can perform merges. Change column names and so on.  
    """

    # Load the uploaded data 
    st.dataframe(data)
    
    st.markdown('## Change datatype')
    col1, col2 = st.columns(2)
    
    name = col1.selectbox("Select Column", data.columns)
    
    column_options = ['numeric','float','object']
    
    type = col2.selectbox("Select Column Type", options=column_options)
    
    if st.button("Change Column Type"):
        if type == 'numeric':
            type = 'int'
        if type == 'object':
            type = 'str'
        
        try:
            data[name] = data[name].astype(type)
            st.write("Your changes have been made!")
        except:
            st.write("Cannot change datatype for this column")
    
    st.markdown('## Select desired column')
    
    options = data.columns
    selected_options = st.multiselect('Which column do you want?',options)
    if len(selected_options) > 0:
        lst = []
        for i in selected_options:
            lst.append(i)
        filtered_df = data[lst]
        st.write(filtered_df)