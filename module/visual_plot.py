from turtle import title
import plotly.graph_objects as go
import streamlit as st

def modelPlot(df):
    df_plot = df.sort_values('score', ascending=False)
    x = df_plot['params'].values
    y = [round(i, 2) for i in df_plot['score'].values]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, 
                         y=y,
                         text=y,
                         textposition='outside'))
    
    fig.update_layout(title="MMM Model",
                      yaxis_range=[min(y)+int(min(y)/1.5), max(y)+int(max(y)/5)])
    
    # trace0 = go.Bar(x=x, 
    #                 y=y)
    
    # data = [trace0]
    # layout = go.Layout(title="Mix MMM Model")
    
    # fig = go.Figure(data=data, layout=layout)
    return fig
