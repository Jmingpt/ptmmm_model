import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO

from module.linearmmm import revenue_linear_mmm, conversion_linear_mmm

#-------------------------------Initialisation-------------------------------#
if 'files' not in st.session_state:
    st.session_state['files'] = None
if 'ga' not in st.session_state:
    st.session_state['ga'] = pd.DataFrame()
if 'fb' not in st.session_state:
    st.session_state['fb'] = pd.DataFrame()
#-------------------------------Initialisation-------------------------------#

def run():
    st.set_page_config(page_title="PT MMM Model", 
                       page_icon="📈", 
                       layout="centered", # centered, wide
                       initial_sidebar_state="auto")
    
    title_col = st.columns((2,2))
    with title_col[0]:
        st.title("MMM Model")
        
    st.caption("Requirements:-")
    st.caption("1. Make sure Facebook data consists of **Date, Campaign name (consists of Remarketing/Prospecting label), Cost, Revenue/Conversions**")
    st.caption("2. Make sure Google data (Google Analytics/Google Ads) consists of **Date, Channel, Cost, Revenue/Conversions**")
    
    # tar_val = st.radio("Targeted Value:", ["Revenue", "Conversions"], horizontal=True)
    
    uploaded_files = st.file_uploader('Upload your files.', type=['csv'], accept_multiple_files=True)
    if len(uploaded_files) == 2:
        with st.expander("Raw Data View:"):
            table_col = st.columns((1,1))
        for f in uploaded_files:
            if 'ga' in str(f.name):
                bytes_data = f.read()
                s = str(bytes_data, 'utf-8')
                data = StringIO(s)
                st.session_state['ga'] = pd.read_csv(data)
                df_ga = st.session_state['ga']
                with table_col[0]:
                    st.write("Google Analytics Data:")
                    st.write(df_ga.sample(5))
                
            elif 'fb' in str(f.name):
                bytes_data = f.read()
                s = str(bytes_data, 'utf-8')
                data = StringIO(s)
                st.session_state['fb'] = pd.read_csv(data)
                df_fb = st.session_state['fb']
                with table_col[1]:
                    st.write("Facebook Ads Data:")
                    st.write(df_fb.sample(5))
        
        if 'Revenue' in df_ga.columns and 'Revenue' in df_fb:
            tar_val = 'Revenue'
        elif 'Conversions' in  df_ga.columns and 'Conversions' in df_fb:
            tar_val = 'Conversions'
        else:
            st.write("Please upload the file with the column of Revenue/Conversions")
                    
        control_col = st.columns((1,1))
        if tar_val == "Revenue":
            with control_col[0]:
                split_ratio = st.number_input("Data Split Ratio:", min_value=0.19, max_value=0.36, value=0.3, step=0.01)
            revenue_linear_mmm(df_ga, df_fb, split_ratio)
        elif tar_val == "Conversions":
            with control_col[0]:
                split_ratio = st.number_input("Data Split Ratio:", min_value=0.19, max_value=0.36, value=0.3, step=0.01)
            conversion_linear_mmm(df_ga, df_fb, split_ratio)
    
    elif len(uploaded_files) == 1:
        st.write('⚠️ Please upload Facebook and Google Data together.')
    
    else:
        st.write('')
        
            

if __name__ == "__main__":
    run()
