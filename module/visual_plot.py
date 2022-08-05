import plotly.graph_objects as go

def modelPlot(df):
    df_plot = df.sort_values('contribution', ascending=False)
    x = df_plot['params'].values
    y = [round(i, 2) for i in df_plot['contribution'].values]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, 
                         y=y,
                         text=y,
                         textposition='outside'))
    
    fig.update_layout(title="MMM Model",
                      yaxis_range=[min(y)-abs(max(y))/2, max(y)+abs(max(y))/2])
    
    # trace0 = go.Bar(x=x, 
    #                 y=y)
    
    # data = [trace0]
    # layout = go.Layout(title="Mix MMM Model")
    
    # fig = go.Figure(data=data, layout=layout)
    return fig