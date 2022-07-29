import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO

from module.linearmmm import mix_linear_mmm, ga_linear_mmm, fb_linear_mmm

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
                    
        # st.subheader("Mixed MMM Result:")
        mix_linear_mmm(df_ga, df_fb, split_ratio)
        
        # with st.expander('Other MMM Model'):
        #     st.subheader("GA MMM Result:")
        #     ga_linear_mmm(df_ga, split_ratio)
        #     st.subheader("FB MMM Result:")
        #     fb_linear_mmm(df_fb, split_ratio)
            

if __name__ == "__main__":
    run()