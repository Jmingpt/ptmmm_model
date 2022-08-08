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
    
    title_col = st.columns((3,1))
    with title_col[0]:
        st.title("PT Market Mix Modeling (MMM)")
        
    st.caption("Requirements:-")
    st.caption("1. Make sure Facebook data consists of **Date, Campaign name, Cost, Revenue/Conversions**")
    st.caption("2. Make sure Google data (Google Analytics/Google Ads) consists of **Date, Channel, Cost, Revenue/Conversions**")

    bdown = st.radio(label="Breakdown Level:-", options=["Channel", "Campaign", "Ad Set", "Ad"], horizontal=True)
    uploaded_files = st.file_uploader('Upload your files.', type=['csv'], accept_multiple_files=True)
    
    if bdown == "Channel":
        process(uploaded_files, channel_revenue, channel_conversions)
        
    else:
        st.subheader("🚧 Under Developing. 🚧")


if __name__ == "__main__":
    run()
