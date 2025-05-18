import streamlit as st
from metricGenerator import generateMetrics
from MonteCarloResampling import resample
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import datetime

st.set_page_config(layout="wide")

df = generateMetrics()
df['date'] = pd.to_datetime(df['date'])
df_index = pd.read_csv("S&P.csv", sep=",")
df_index['Date'] = pd.to_datetime(df_index['Date'])
df_index = df_index.loc[df_index['Date'] >= datetime.datetime(2017, 3, 1)]
df_index[' Close'] = df_index[' Close'] / df_index[' Close'].values[-1]


df_comparison = df.merge(df_index[['Date', ' Close']], how='left', left_on='date', right_on='Date').dropna()
df_comparison['r_port'] = df_comparison['equity'].pct_change()
df_comparison['r_index'] = df_comparison[' Close'].pct_change()
df_comparison = df_comparison.dropna()

covariance = df_comparison[['r_port', 'r_index']].cov().iloc[0, 1]
index_variance = df_comparison['r_index'].var()
beta = covariance / index_variance

#st.table(df_comparison)

risk_free_rate = 0.0245
risk_free_daily = (1+risk_free_rate)**(1/252)- 1

market_return = (df_comparison[' Close'].values[-1] - 1)/1
portfolio_return = (df_comparison['equity'].values[-1] - 1)/1

expectedReturns = risk_free_rate + beta*(market_return-risk_free_rate)

df_comparison['strategy_return'] = df_comparison['equity'].pct_change()

sharpe_ratio = (portfolio_return - risk_free_rate)/((df_comparison['strategy_return']-risk_free_daily).var())**(1/2)
print(((df_comparison['strategy_return']-risk_free_daily).var())**(1/2))

df_comparison['excess_return'] = df_comparison['strategy_return'] - risk_free_daily

sharpe_daily = df_comparison['excess_return'].mean() / df_comparison['strategy_return'].std()
sharpe_annualized = sharpe_daily * (252 ** 0.5)



#st.write(market_return)
#st.write(portfolio_return)
#st.write(expectedReturns)

alpha = portfolio_return - expectedReturns


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
        x=pd.to_datetime(df_comparison['date']),
        y=df_comparison['equity'],
        mode='lines',
        name='Equity Curve',
        line=dict(color='blue')
    ), row=1, col=1)

    fig.append_trace(go.Scatter(
        x=pd.to_datetime(df_comparison['Date']),
        y=df_comparison[' Close'],
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


with st.expander("Strategy description and bias handling"):
    st.markdown("""
    **Look-Ahead Bias** shoul not be present. The system trades on the open and close of day T, based on information from day T-1 and previous days.         

    **Survivorship Bias** remains challanging. The dataset contains both failed and survining companies in the S&P 500, but for a limited number of years. The strategy is set up to only trade if there is
                at least 3 years of trading data, and it is possible that many failed companies are not included. 
    """)


col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
        <div style="border: 2px solid black; border-radius: 8px; padding: 20px; height: 200px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center; display: flex; flex-direction: column; justify-content: center; align-items: center;">
            <div style="font-weight: bold; font-size: 24px; margin-bottom: 10px;">Alpha</div>
            <div style="font-size: 22px;">{100*alpha:.0f}%</div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
        <div style="border: 2px solid black; border-radius: 8px; padding: 20px; height: 200px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center; display: flex; flex-direction: column; justify-content: center; align-items: center;">
            <div style="font-weight: bold; font-size: 24px; margin-bottom: 10px;">Beta</div>
            <div style="font-size: 22px;">{beta:.2f}</div>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
        <div style="border: 2px solid black; border-radius: 8px; padding: 20px; height: 200px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center; display: flex; flex-direction: column; justify-content: center; align-items: center;">
            <div style="font-weight: bold; font-size: 24px; margin-bottom: 10px;">Sharpe ratio</div>
            <div style="font-size: 22px;">{sharpe_annualized:.2f}</div>
        </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
        <div style="border: 2px solid black; border-radius: 8px; padding: 20px; height: 200px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center; display: flex; flex-direction: column; justify-content: center; align-items: center;">
            <div style="font-weight: bold; font-size: 24px; margin-bottom: 10px;">Assumed trading costs</div>
            <div style="font-size: 22px;">Slippage = 0.05% <br> Fee = 0.02% per trade</div>
        </div>
    """, unsafe_allow_html=True)



fig_mc = resample()
fig_mc = fig_mc.update_layout(
        title='Monte Carlo resampling',
        xaxis_title='Time',
        yaxis_title='Equity',
        showlegend=False,
        autosize=True  # Make the plot auto resize
    )

st.plotly_chart(fig_mc, use_container_width=True)
