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
    with title_col[1]:
        split_ratio = st.number_input("Split Ratio:", min_value=0.19, max_value=0.36, value=0.3, step=0.01)
        
    st.caption("Requirements:-")
    st.caption("1. Make sure Facebook data consists of **Date, Campaign name, Cost, Revenue/Conversions**")
    st.caption("2. Make sure Google data (Google Analytics/Google Ads) consists of **Date, Channel, Cost, Revenue/Conversions**")
    st.caption("3. Select targeted value (Revenue/Conversions) format.")
    
    tar_val = st.radio("Targeted Value:", ["Revenue", "Conversions"], horizontal=True)
    
    uploaded_files = st.file_uploader('Upload your files.', type=['csv'], accept_multiple_files=True)
    if uploaded_files:
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
                    
        if tar_val == "Revenue":
            revenue_linear_mmm(df_ga, df_fb, split_ratio)
        elif tar_val == "Conversions":
            conversion_linear_mmm(df_ga, df_fb, split_ratio)
        
            

if __name__ == "__main__":
    run()
