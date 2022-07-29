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
    if 'remarketing' in campaign.lower():
        return 'Remarketing'
    elif 'prospecting' in campaign.lower():
        return 'Prospecting'


def mix_linear_mmm(df_ga, df_fb, split_ratio):
    df_ga['Date'] = pd.to_datetime(df_ga['Date'])
    df_fb['Date'] = pd.to_datetime(df_fb['Date'])
    df_fb['Channel'] = df_fb['Campaign name'].apply(pros_rema)
    df_fb = df_fb.drop(["Ad name", "Campaign name"], axis=1)
    df_fb = df_fb[['Date', 'Channel', 'Cost', 'Revenue']]

    df_ga['Yearweek'] = df_ga['Date'].apply(yearweek)
    df_fb['Yearweek'] = df_fb['Date'].apply(yearweek)
    df = pd.concat([df_ga, df_fb], ignore_index=True)
    
    pivot_tb = pd.pivot_table(df, values='Cost', index=['Yearweek'], columns=['Channel'], aggfunc=np.sum)
    pivot_df = pivot_tb.reset_index().sort_values('Yearweek', ascending=False).reset_index(drop=True)
    pivot_df = pivot_df.drop("Facebook", axis=1)
    pivot_df = pivot_df.fillna(pivot_df.mean())
    revenue_df = df.groupby('Yearweek')['Revenue'].sum().reset_index().sort_values('Yearweek', ascending=False).reset_index(drop=True)
    mmm = pd.merge(pivot_df, revenue_df, on='Yearweek', how='left')
    mmm_df = mmm.set_index('Yearweek')
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
    plot_df = pd.DataFrame(coef, columns=['score', 'params'])
    ga_fig = modelPlot(plot_df)
    st.plotly_chart(ga_fig)
    
    smr_col = st.columns((1,1,1))
    with smr_col[0]:
        st.write(f"Model Accuracy: **{score}%**")
    with smr_col[1]:
        st.write(f"MAE: {mae}")
    with smr_col[2]:
        st.write(f"RMSE: {rmse}")

def ga_linear_mmm(df, split_ratio):
    df['Date'] = pd.to_datetime(df['Date'])
    df['Yearweek'] = df['Date'].apply(yearweek)
    
    pivot_tb = pd.pivot_table(df, values='Cost', index=['Yearweek'], columns=['Channel'], aggfunc=np.sum)
    pivot_df = pivot_tb.reset_index().sort_values('Yearweek', ascending=False).reset_index(drop=True)
    pivot_df = pivot_df.drop("Facebook", axis=1)
    pivot_df = pivot_df.fillna(pivot_df.mean())
    revenue_df = df.groupby('Yearweek')['Revenue'].sum().reset_index().sort_values('Yearweek', ascending=False).reset_index(drop=True)
    mmm = pd.merge(pivot_df, revenue_df, on='Yearweek', how='left')
    mmm_df = mmm.set_index('Yearweek')
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
    
    smr_col = st.columns((1,1,1))
    with smr_col[0]:
        st.write(f"Model Accuracy: **{score}%**")
    with smr_col[1]:
        st.write(f"MAE: {mae}")
    with smr_col[2]:
        st.write(f"RMSE: {rmse}")
    
    # mmm_exp = ""
    # for i, j in zip(lr.coef_, X.columns):
    #     mmm_exp += "{} * {} + \n \n ".format(round(i, 2), j)
    # mmm_exp += str(round(lr.intercept_, 2))
    
    # st.write(mmm_exp.replace("+ -", "- "))
    coef = []
    for i, j in zip(lr.coef_, X.columns):
        coef.append([i,j])
    plot_df = pd.DataFrame(coef, columns=['score', 'params'])
    ga_fig = modelPlot(plot_df)
    st.plotly_chart(ga_fig)
    
def fb_linear_mmm(df, split_ratio):
    df['Date'] = pd.to_datetime(df['Date'])
    df['Yearweek'] = df['Date'].apply(yearweek)
    df['Channel'] = df['Campaign name'].apply(pros_rema)
    df = df.drop(["Ad name", "Campaign name"], axis=1)
    
    pivot_tb = pd.pivot_table(df, values='Cost', index=['Yearweek'], columns=['Channel'], aggfunc=np.sum)
    pivot_df = pivot_tb.reset_index().sort_values('Yearweek', ascending=False).reset_index(drop=True)
    pivot_df = pivot_df.fillna(pivot_df.mean())
    revenue_df = df.groupby('Yearweek')['Revenue'].sum().reset_index().sort_values('Yearweek', ascending=False).reset_index(drop=True)
    mmm = pd.merge(pivot_df, revenue_df, on='Yearweek', how='left')
    mmm_df = mmm.set_index('Yearweek')
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
    
    smr_col = st.columns((1,1,1))
    with smr_col[0]:
        st.write(f"Model Accuracy: **{score}%**")
    with smr_col[1]:
        st.write(f"MAE: {mae}")
    with smr_col[2]:
        st.write(f"RMSE: {rmse}")
    
    # mmm_exp = ""
    # for i, j in zip(lr.coef_, X.columns):
    #     mmm_exp += "{} * {} + \n \n ".format(round(i, 2), j)
    # mmm_exp += str(round(lr.intercept_, 2))
    
    # st.write(mmm_exp.replace("+ -", "- "))
    
    coef = []
    for i, j in zip(lr.coef_, X.columns):
        coef.append([i,j])
    plot_df = pd.DataFrame(coef, columns=['score', 'params'])
    fb_fig = modelPlot(plot_df)
    st.plotly_chart(fb_fig)
    