import pandas as pd
import numpy as np
import time
import plotly.express as px

def resample():
    df_equity = pd.read_csv('equity_curve.csv')
    df_equity['return'] = ((df_equity['equity'] - df_equity['equity'].shift(1))/df_equity['equity']).fillna(0)
    #rng = np.random.default_rng()  # creates a new generator instance with system entropy
    for i in range(300):
        np.random.seed(int(time.time() * 1000) % 2**32)
        df_equity['return shuffled'] = np.random.permutation(df_equity['return'].values)
        print(df_equity['return shuffled'].values[0])
        df_equity[f'Reshuffled_{i}'] = (1 + df_equity['return shuffled']).cumprod()


    df_equity = df_equity.drop(["return shuffled", "return"], axis=1)
    df_equity = df_equity.set_index("date")
    print(df_equity)


    fig = px.line(df_equity)
    return fig

resample()

