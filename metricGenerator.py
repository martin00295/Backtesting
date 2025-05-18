import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generateMetrics():
    df = pd.read_csv('equity_curve.csv')
    print(df)

    running_max = df['equity'].cummax()

    # 2. Calculate drawdown
    df['rel_drawdown'] = (df['equity'] - running_max) / running_max

    '''print(df)


 # Create subplots: 2 rows, 1 column
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True,  # Share x-axis for both plots
        vertical_spacing=0.1,  # Space between plots
        row_heights=[0.7, 0.3],  # Adjust row heights (more space for the equity curve)
        subplot_titles=("Equity Curve", "Drawdown (%)"),
        shared_yaxes=True
    )

    # Add equity curve to the first subplot
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['equity'],
        mode='lines',
        name='Equity Curve',
        line=dict(color='blue')
    ), row=1, col=1)

    # Add drawdown as bars to the second subplot
    fig.add_trace(go.Bar(
        x=df.index,
        y=df['rel_drawdown'] * 100,  # Convert to percentage
        name='Drawdown (%)',
        marker=dict(color='red')
    ), row=2, col=1)

    # Update layout
    fig.update_layout(
        title='Equity Curve and Drawdown',
        xaxis_title='Time',
        yaxis_title='Equity',
        template='plotly_dark',
        showlegend=True
    )

    # Show the figure
    fig.show()'''

    return df


if __name__ == '__main__':
    generateMetrics()