import pandas as pd
import streamlit as st 
import os

def meta_data(data):
    """This application is created to help the user change the metadata for the uploaded file. 
    They can perform merges. Change column names and so on.  
    """

    # Load the uploaded data 
    st.dataframe(data)
    # df = data.astype(str)
    # st.write(data.dtypes)
    # column_name = data.columns
    # # Read the column meta data for this dataset 
    # pd.DataFrame(data.c)
    # col_metadata = pd.concat([data.dtypes,data.columns],,axis=1)
    # 
    # ''' Change the information about column types
    #     Here the info of the column types can be changed using dropdowns.
    #     The page is divided into two columns using beta columns 
    # '''
    # st.markdown("#### Change the information about column types")
    # 
    # # Use two column technique 
    col1, col2 = st.columns(2)
    # 
    # global name, type
    # Design column 1 
    name = col1.selectbox("Select Column", data.columns)
    
    # Design column two 
    # current_type = col_metadata[col_metadata['column_name'] == name]['type'].values[0]
    # print(data.dtypes)
    column_options = ['numeric','float','object']
    # # current_index = column_options.index(current_type)
    # 
    type = col2.selectbox("Select Column Type", options=column_options)
    # 
    # st.write("""Select your column name and the new type from the data.
    #             To submit all the changes, click on *Submit changes* """)
    # 
    # 
    if st.button("Change Column Type"):
        if type == 'numeric':
            type = 'int'
        if type == 'object':
            type = 'str'
        data[name] = data[name].astype(type)
        st.write("Your changes have been made!")
        st.write(data.dtypes)
    # 
    #     # Set the value in the metadata and resave the file 
    #     # col_metadata = pd.read_csv('data/metadata/column_type_desc.csv')
    #     st.dataframe(col_metadata[col_metadata['column_name'] == name])
    # 
    #     col_metadata.loc[col_metadata['column_name'] == name, 'type'] = type
    #     col_metadata.to_csv('data/metadata/column_type_desc.csv', index = False)
    # 
    #     st.write("Your changes have been made!")
    #     st.dataframe(col_metadata[col_metadata['column_name'] == name])