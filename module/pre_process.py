import streamlit as st
import pandas as pd
from io import StringIO

def process(uploaded_files, revenue_mmm, conversions_mmm):
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
                split_ratio = st.number_input(label="Data Split Ratio:", 
                                            min_value=0.15, 
                                            max_value=0.4, 
                                            value=0.3, 
                                            step=0.01)
            revenue_mmm(df_ga, df_fb, split_ratio)
        elif tar_val == "Conversions":
            with control_col[0]:
                split_ratio = st.number_input("Data Split Ratio:", 
                                            min_value=0.15, 
                                            max_value=0.4, 
                                            value=0.3, 
                                            step=0.01)
            conversions_mmm(df_ga, df_fb, split_ratio)
    
    elif len(uploaded_files) == 1:
        st.write('⚠️ Please upload Facebook and Google data together.')
    
    else:
        st.write('')