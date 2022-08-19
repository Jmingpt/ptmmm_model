import streamlit as st
import pandas as pd

from module.pre_process import process
from module.linearmmm import channel_revenue, channel_conversions, campaign_revenue, campaign_conversions, adset_revenue, adset_conversions, adformat_revenue, adformat_conversions

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
                       layout="wide", # centered, wide
                       initial_sidebar_state="auto")
    
    st.title("PT Market Mix Modelling (MMM)")
    st.markdown("What is Market Mix Modelling (MMM)? https://towardsdatascience.com/market-mix-modeling-mmm-101-3d094df976f9")
    
    with st.expander("❗️Requirements:- (Read before using the tool.)", expanded=False):
        st.caption("1. Only **_.csv_** file is accepted. File name should consist of **'fb'/'Facebook'** or **'ga'/'Google'**")
        st.caption("2. Make sure Facebook data consists of **_Date, Campaign name, Cost, Revenue/Conversions._** Please rename the columns before uploading.")
        st.caption("3. Make sure Google data (Google Analytics/Google Ads) consists of **_Date, Channel, Campaign, Cost, Revenue/Conversions._** Please rename the columns before uploading.")
        st.caption("4. Recommended date range: **_1 year._**")

    bdown = st.radio(label="Breakdown Level:-", options=["Channel", "Campaign", "Ad Set", "Ad Format"], horizontal=True)
    uploaded_files = st.file_uploader('Upload your files.', type=['csv'], accept_multiple_files=True)
    
    if bdown == "Channel":
        process(uploaded_files, channel_revenue, channel_conversions)
        
    elif bdown == 'Campaign':
        process(uploaded_files, campaign_revenue, campaign_conversions)
    
    elif bdown == 'Ad Set':
        process(uploaded_files, adset_revenue, adset_conversions)
        
    elif bdown == 'Ad Format':
        process(uploaded_files, adformat_revenue, adformat_conversions)
    
    else:
        st.subheader("🚧 Under Construction 🚧")


if __name__ == "__main__":
    run()
