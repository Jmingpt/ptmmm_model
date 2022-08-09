import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from .visual_plot import modelPlot

def yearweek(dtime):
    year = dtime.isocalendar()[0]
    week = dtime.isocalendar()[1]
    yw = "{}{:02d}".format(year, week)

    return yw

def pros_rema(campaign):
    if campaign:
        return 'Facebook'


def channel_revenue(df_ga, df_fb, split_ratio):
    df_ga['Date'] = pd.to_datetime(df_ga['Date'])
    df_fb['Date'] = pd.to_datetime(df_fb['Date'])
    df_ga = df_ga[df_ga['Channel'].notna()]
    df_ga = df_ga[df_ga['Channel'] != 'Facebook']
    df_fb['Channel'] = df_fb['Campaign name'].apply(pros_rema)
    df_ga = df_ga[['Date', 'Channel', 'Cost', 'Revenue']]
    df_fb = df_fb[['Date', 'Channel', 'Cost', 'Revenue']]

    df = pd.concat([df_ga, df_fb], ignore_index=True)
    # df = df[df['Channel'] != 'Facebook']
    
    pivot_tb = pd.pivot_table(df, values='Cost', index=['Date'], columns=['Channel'], aggfunc=np.sum)
    pivot_df = pivot_tb.reset_index().sort_values('Date', ascending=False).reset_index(drop=True)
    # pivot_df = pivot_df.fillna(pivot_df.mean())
    pivot_df = pivot_df.fillna(0)
    revenue_df = df.groupby('Date')['Revenue'].sum().reset_index().sort_values('Date', ascending=False).reset_index(drop=True)
    mmm = pd.merge(pivot_df, revenue_df, on='Date', how='left')
    mmm_df = mmm.set_index('Date')
    X = mmm_df.drop(columns=['Revenue'])
    y = mmm_df['Revenue']
    
    lst = []
    lr = LinearRegression()
    for i in range(500):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_ratio, random_state=i)
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)
        lst.append([i, round(r2_score(y_test, y_pred), 4)])
        
    op_random_state = pd.DataFrame(lst, columns=['idx', 'score']).set_index('idx').idxmax().values[0]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_ratio, random_state=op_random_state)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    
    score=round(100*r2_score(y_test, y_pred), 2)
    mae=round(mean_absolute_error(y_test, y_pred), 2)
    rmse=round(np.sqrt(mean_squared_error(y_test, y_pred)), 2)
    
    coef = []
    for i, j in zip(lr.coef_, X.columns):
        coef.append([i,j])
    plot_df = pd.DataFrame(coef, columns=['coef', 'params'])
    plot_df['mean_input'] = X_test.mean().values
    plot_df['contribution'] = plot_df['coef']*plot_df['mean_input']
    ga_fig = modelPlot(plot_df)
    st.plotly_chart(ga_fig, use_container_width=True)
    
    smr_col = st.columns((1,1,1))
    with smr_col[0]:
        st.write(f"Model Accuracy: **{score}%**")
    with smr_col[1]:
        st.write(f"MAE: {mae}")
    with smr_col[2]:
        st.write(f"RMSE: {rmse}")

def channel_conversions(df_ga, df_fb, split_ratio):
    df_ga = df_ga[df_ga['Date'].notna()]
    df_fb = df_fb[df_fb['Date'].notna()]
    df_ga['Date'] = pd.to_datetime(df_ga['Date'])
    df_fb['Date'] = pd.to_datetime(df_fb['Date'])
    df_ga = df_ga[df_ga['Channel'].notna()]
    df_ga = df_ga[df_ga['Channel'] != 'Facebook']
    df_fb['Channel'] = df_fb['Campaign name'].apply(pros_rema)
    df_ga = df_ga[['Date', 'Channel', 'Cost', 'Conversions']]
    df_fb = df_fb[['Date', 'Channel', 'Cost', 'Conversions']]

    df = pd.concat([df_ga, df_fb], ignore_index=True)
    
    pivot_tb = pd.pivot_table(df, values='Cost', index=['Date'], columns=['Channel'], aggfunc=np.sum)
    pivot_df = pivot_tb.reset_index().sort_values('Date', ascending=False).reset_index(drop=True)
    pivot_df = pivot_df.fillna(0)
    revenue_df = df.groupby('Date')['Conversions'].sum().reset_index().sort_values('Date', ascending=False).reset_index(drop=True)
    mmm = pd.merge(pivot_df, revenue_df, on='Date', how='left')
    mmm_df = mmm.set_index('Date')
    X = mmm_df.drop(columns=['Conversions'])
    y = mmm_df['Conversions']
    
    lst = []
    lr = LinearRegression()
    for i in range(500):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_ratio, random_state=i)
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)
        lst.append([i, round(r2_score(y_test, y_pred), 4)])
        
    op_random_state = pd.DataFrame(lst, columns=['idx', 'score']).set_index('idx').idxmax().values[0]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_ratio, random_state=op_random_state)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    
    score=round(100*r2_score(y_test, y_pred), 2)
    mae=round(mean_absolute_error(y_test, y_pred), 2)
    rmse=round(np.sqrt(mean_squared_error(y_test, y_pred)), 2)
    
    coef = []
    for i, j in zip(lr.coef_, X.columns):
        coef.append([i,j])
    plot_df = pd.DataFrame(coef, columns=['coef', 'params'])
    plot_df['mean_input'] = X_test.mean().values
    plot_df['contribution'] = plot_df['coef']*plot_df['mean_input']
    ga_fig = modelPlot(plot_df)
    st.plotly_chart(ga_fig)
    
    smr_col = st.columns((1,1,1))
    with smr_col[0]:
        st.write(f"Model Accuracy: **{score}%**")
    with smr_col[1]:
        st.write(f"MAE: {mae}")
    with smr_col[2]:
        st.write(f"RMSE: {rmse}")
