import streamlit as st
import pandas as pd

from module.pre_process import process
from module.linearmmm import channel_revenue, channel_conversions

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
    
    st.title("PT Market Mix Modeling (MMM)")
    with st.expander("Requirements:- (Read before using the tool.)", expanded=False):
        st.caption("1. Only **.csv** file is accepted.")
        st.caption("2. Make sure Facebook data consists of **_Date, Campaign name, Cost, Revenue/Conversions_.** Please rename the columns before uploading.")
        st.caption("3. Make sure Google data (Google Analytics/Google Ads) consists of **_Date, Channel, Cost, Revenue/Conversions_.** Please rename the columns before uploading.")
        st.caption("4. Recommended date range: **1 year.**")

    bdown = st.radio(label="Breakdown Level:-", options=["Channel", "Campaign", "Ad Set", "Ad"], horizontal=True)
    uploaded_files = st.file_uploader('Upload your files.', type=['csv'], accept_multiple_files=True)
    
    if bdown == "Channel":
        process(uploaded_files, channel_revenue, channel_conversions)
        
    else:
        st.subheader("🚧 Under Construction 🚧")


if __name__ == "__main__":
    run()
