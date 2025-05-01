import streamlit as st
from metricGenerator import generateMetrics
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import datetime

st.set_page_config(layout="wide")

df = generateMetrics()
df_index = pd.read_csv("S&P.csv", sep=",")
df_index['Date'] = pd.to_datetime(df_index['Date'])
df_index = df_index.loc[df_index['Date'] >= datetime.datetime(2017, 3, 1)]
df_index[' Close'] = df_index[' Close'] / df_index[' Close'].values[-1]


print(df_index)

def createFig(df):
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True,  # Share x-axis for both plots
        vertical_spacing=0.1,  # Space between plots
        row_heights=[0.7, 0.3],  # Adjust row heights (more space for the equity curve)
        subplot_titles=("Equity Curve", "Drawdown (%)"),
        shared_yaxes=True
    )

    # Add equity curve to the first subplot
    fig.append_trace(go.Scatter(
        x=pd.to_datetime(df['date']),
        y=df['equity'],
        mode='lines',
        name='Equity Curve',
        line=dict(color='blue')
    ), row=1, col=1)

    fig.append_trace(go.Scatter(
        x=pd.to_datetime(df_index['Date']),
        y=df_index[' Close'],
        mode='lines',
        name='Index (S&P 500)',
        line=dict(color='grey')
    ), row=1, col=1)

    # Add drawdown as bars to the second subplot
    fig.add_trace(go.Bar(
        x=df['date'],
        y=df['rel_drawdown'] * 100,  # Convert to percentage
        name='Drawdown (%)',
        marker=dict(color='red')
    ), row=2, col=1)

    # Update layout
    fig.update_layout(
        title='Walk forward backtest',
        xaxis_title='Time',
        yaxis_title='Equity',
        template='plotly_dark',
        showlegend=True,
        autosize=True  # Make the plot auto resize
    )

    return fig

st.title('Machine learning quantitative trading system')
st.plotly_chart(createFig(df), use_container_width=True)


with st.expander("Bias handling"):
    st.markdown("""
    **Look-Ahead Bias** is not present. The system trades on the open and close of day T, based on information from day T-1 and previous days.         

    **Survivorship Bias** remains challanging. The dataset contains both failed and survining companies in the S&P 500, but since
    the history is limited, most of the failed companies are not traded by the system.
    """)