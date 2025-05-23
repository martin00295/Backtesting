import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


slippage_pct = 0.0005       # 0.05% slippage
transaction_fee_pct = 0.0002  # 0.02% per trade (entry and exit)


def predict(df, date, target='Close', verbose=False, useYFinance=False):
    try:
        df = df.drop(['OpenInt'])
    except Exception:
        pass
    df_train = df.loc[df.index < date]
    df_test = df.loc[df.index == date]
   
    if len(df_test) == 0:
        return -10000
    
    if len(df_train) < 4*253: # 3 years of business days
        return -10000
    
    df_train = df_train

    cutoff_date = date - pd.DateOffset(years=3)
    df_train = df[df.index > cutoff_date]
    
    y_train = df_train[target]
    X_train = df_train.drop(['Open', 'High', 'Low', 'Close', 'Volume', 
        'log_rtn', 'in_log_rtn', 'sigma_30', 'sigma_7', 'DeltaV', 'mu_30', 'mu_7', 'Open_z', 'intraday_log_rtn'], axis=1)
    y_test = df_test[target]
    X_test = df_test.drop(['Open', 'High', 'Low', 'Close', 'Volume', 
        'log_rtn', 'in_log_rtn', 'sigma_30', 'sigma_7', 'DeltaV', 'mu_30', 'mu_7', 'Open_z', 'intraday_log_rtn'], axis=1)
    

    #print(X_test.columns)
    

    model = Pipeline([
    ('normalizer', StandardScaler()),
    ('regressor', LinearRegression())
    ])

    model.fit(X_train, y_train)
    result = model.predict(X_test)
    

    if verbose:
        print(f"True value: {y_test.values[0]}")
        print(f"Predicted value: {round(result[0])}")
        print(f"Percentage missed: {round(100*np.abs((result[0]-y_test.values[0])/y_test.values[0]),2)}%")

    return result[0]


def yFinanceAddFeatures(df, historySteps = 2):
    df['log_rtn'] = np.log(df['Close']).diff()
    df['intraday_log_rtn'] = np.log(df['Close'])-np.log(df['Open'])

    df['DeltaV'] = df['Volume'] - df['Volume'].shift(1)
    df['log_rtn'] = np.log(df['Close']).diff() # Log returns over day
    df['in_log_rtn'] = np.log(df['Close'])-np.log(df['Open']) # Intraday log return

    df['sigma_30'] = df['log_rtn'].rolling(window=30).std(ddof=1) * np.sqrt(252)
    df['sigma_7'] = df['log_rtn'].rolling(window=7).std(ddof=1) * np.sqrt(252)

    df['mu_30'] = df['log_rtn'].rolling(window=30).mean() * np.sqrt(252)
    df['mu_7'] = df['log_rtn'].rolling(window=7).mean() * np.sqrt(252)

    df['Open_z'] = (df['Open'] - df['mu_30'])/df['sigma_30']

    # Add closing price of previous days
    for i in range(1, historySteps):
        df[f'Close T-{i}'] = df['Close'].shift(i)

    # Add open price of previous days
    for i in range(1, historySteps):
        df[f'Open T-{i}'] = df['Open'].shift(i)

    # Add high price of previous days
    for i in range(1, historySteps):
        df[f'High T-{i}'] = df['High'].shift(i)

    # Add low price of previous days
    for i in range(1, historySteps):
        df[f'Low T-{i}'] = df['Low'].shift(i)

    # Add log return of previous days
    for i in range(1, historySteps):
        df[f'log_rtn T-{i}'] = df['log_rtn'].shift(i)

    # Add intraday log return of previous days
    for i in range(1, historySteps):
        df[f'in_log_rtn T-{i}'] = df['in_log_rtn'].shift(i)

    # Add volatility of previous days
    for i in range(1, historySteps):
        df[f'sigma_30 T-{i}'] = df['sigma_30'].shift(i)
    for i in range(1, historySteps):
        df[f'sigma_7 T-{i}'] = df['sigma_7'].shift(i)

    df = df.dropna()
    return df

def fetchData(file = 'Stocks/abb.us.txt', historySteps = 2):
    df = None
    try:
        df = pd.read_csv(file, index_col='Date')
    except:
        pass
    try:
        df = pd.read_csv(file, index_col='date')
    except:
        pass

    if df is None:
        print(f'File not correct: {file}')

    df.index = pd.to_datetime(df.index)

    df.columns = df.columns.str.capitalize()

    df['log_rtn'] = np.log(df['Close']).diff()
    df['intraday_log_rtn'] = np.log(df['Close'])-np.log(df['Open'])

    df['DeltaV'] = df['Volume'] - df['Volume'].shift(1)
    df['log_rtn'] = np.log(df['Close']).diff() # Log returns over day
    df['in_log_rtn'] = np.log(df['Close'])-np.log(df['Open']) # Intraday log return

    df['sigma_30'] = df['log_rtn'].rolling(window=30).std(ddof=1) * np.sqrt(252)
    df['sigma_7'] = df['log_rtn'].rolling(window=7).std(ddof=1) * np.sqrt(252)

    df['mu_30'] = df['log_rtn'].rolling(window=30).mean() * np.sqrt(252)
    df['mu_7'] = df['log_rtn'].rolling(window=7).mean() * np.sqrt(252)

    df['Open_z'] = (df['Open'] - df['mu_30'])/df['sigma_30']

    # Add closing price of previous days
    for i in range(1, historySteps):
        df[f'Close T-{i}'] = df['Close'].shift(i)

    # Add open price of previous days
    for i in range(1, historySteps):
        df[f'Open T-{i}'] = df['Open'].shift(i)

    # Add high price of previous days
    for i in range(1, historySteps):
        df[f'High T-{i}'] = df['High'].shift(i)

    # Add low price of previous days
    for i in range(1, historySteps):
        df[f'Low T-{i}'] = df['Low'].shift(i)

    # Add log return of previous days
    for i in range(1, historySteps):
        df[f'log_rtn T-{i}'] = df['log_rtn'].shift(i)

    # Add intraday log return of previous days
    for i in range(1, historySteps):
        df[f'in_log_rtn T-{i}'] = df['in_log_rtn'].shift(i)

    # Add volatility of previous days
    for i in range(1, historySteps):
        df[f'sigma_30 T-{i}'] = df['sigma_30'].shift(i)
    for i in range(1, historySteps):
        df[f'sigma_7 T-{i}'] = df['sigma_7'].shift(i)

    df = df.dropna()

    return df
    



# Generates a signal of whether to perform the trade or not
def generateSignal(daysToPredict=365*1, useYFinance=False, writeSignals=True, readSignals=False):

    if useYFinance:
        df_yfinance = pd.read_csv("2015-2025.csv")
        df_yfinance['Date'] = pd.to_datetime(df_yfinance['Date'])
        df_yfinance = df_yfinance.set_index('Date')
        folder = 'Data/data_not_bias'
        filenames = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.csv')][200:]

        dfs = {filename :yFinanceAddFeatures(df_yfinance[[ j+filename.split("/")[-1].split(".")[-2].upper() for j in ["Close_", "Open_", "Volume_", "High_", "Low_"]]].rename(columns={
            "Close_"+filename.split("/")[-1].split(".")[-2].upper(): 'Close',
            "Open_"+filename.split("/")[-1].split(".")[-2].upper(): 'Open',
            "Volume_"+filename.split("/")[-1].split(".")[-2].upper(): 'Volume',
            "High_"+filename.split("/")[-1].split(".")[-2].upper(): 'High',
            "Low_"+filename.split("/")[-1].split(".")[-2].upper(): 'Low'
            })) for filename in filenames}
        #print(dfs)
    else:

        #filenames = ['Stocks/abb.us.txt', 'Stocks/abbv.us.txt', 'Stocks/abc.us.txt', 'Stocks/abcb.us.txt'
        #            ]
        #folder = 'Stocks'
        #filenames = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.txt')][0:30]
        folder = 'Data/data_not_bias'
        filenames = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.csv')][100:200]



        dfs = {filename : fetchData(filename) for filename in filenames}

    signals = {}

    #dates = dfs['Stocks/abb.us.txt'].index[-daysToPredict:]
    #print(dates)
    #print()
    endDate = dfs[filenames[0]].index[-1:][0]
    dates = pd.date_range(end=endDate, periods=daysToPredict, freq='B')  # 'B' = business day

    print(f'Prediciting from {dates[0]} to {dates[-1]}.')

    print(f' --- Starting by generating signal ---')

    # Generate signals
    if readSignals:
        signals = pd.read_csv('signals100.csv')
    elif True:
        iterationCounter = 1
        iterationToPerform = len(dfs.keys())
        for key, value in dfs.items():
            print(f'generating signal for equity {iterationCounter} of {iterationToPerform}')
            print()
            iterationCounter += 1

            df = value

            predictionList = []
            #for date in df.index[-daysToPredict:]:
            for date in dates:
                predictionList.append(predict(df, date, 'in_log_rtn'))
            
            signals[key] = predictionList

    if writeSignals:
        df = pd.DataFrame(signals)
        # Save the DataFrame to CSV
        df.to_csv('signals.csv', index=False)

    print(f'--- Performing allocation ---')


    # Find relative allocations
    def keep_n_largest(lst, n):
        # Get the indices of the n largest non-zero elements
        non_zero_indices = [i for i, val in enumerate(lst) if val > 0]
        if len(non_zero_indices) <= n:
            return lst.copy()

        # Find the n largest values and their indices
        top_n_indices = sorted(non_zero_indices, key=lambda i: lst[i], reverse=True)[:n]

        # Build new list, zeroing out the rest
        return [val if i in top_n_indices else 0 for i, val in enumerate(lst)]
    
    allocatedSharesTotal = np.zeros(len(filenames), dtype=float)
    lst_relative_allocations = []
    for i in range(len(dates)):
        signals_day = []
        for key, value in signals.items():
            signals_day.append(value[i])
        signals_day = [i if i > 0.001 else 0 for i in signals_day]
        #signals_day = [i if i > -0.0000 else -1000 for i in signals_day]
        signals_day = keep_n_largest(signals_day, 5) # Maximum 5 trades per day to limit commissions
        #signals_day = [i if i > 0.000 else 0 for i in signals_day]
        if np.sum(signals_day) > 0:
            relativeAllocation = np.array(signals_day)/np.sum(signals_day)
            assert(np.abs(np.sum(relativeAllocation)-1)<10**(-9))
            lst_relative_allocations.append(relativeAllocation)
            allocatedSharesTotal += np.array(relativeAllocation)
        else:
            lst_relative_allocations.append(None)





    ### Perform trades
    equityStart = 1
    equityStart_no_fee = 1
    equityCurve = []
    equityCurve_no_fee = []
    for i in range(len(dates)):
        date = dates[i]
        relativeAllocation = lst_relative_allocations[i]
        


        if relativeAllocation is None:
            equityEnd = equityStart
            equityEnd_no_fee = equityStart
        else:
            equityEnd = 0
            equityEnd_no_fee = 0
            for j, (asset, df) in enumerate(dfs.items()):
                allocationShare = relativeAllocation[j]
                if not allocationShare > 0: continue

                open = df.loc[df.index == date]['Open'].values[0]
                close = df.loc[df.index == date]['Close'].values[0]

                entry_price = open * (1 + slippage_pct)
                exit_price = close * (1 - slippage_pct)

                equityEnd += equityStart * allocationShare * (exit_price/entry_price - 2*transaction_fee_pct)
                equityEnd_no_fee += equityStart_no_fee * allocationShare * (close/open)


        equityCurve.append(equityEnd)
        equityCurve_no_fee.append(equityEnd_no_fee)
        equityStart = equityEnd
        equityStart_no_fee = equityEnd_no_fee

    
    #print(equityCurve)
    nTrades = 0
    for value in lst_relative_allocations:
        if not value is None:
            trades = (value > 0).sum()
            nTrades += trades
    tradesPerDay = nTrades/daysToPredict
    print(f"Trades per day: {tradesPerDay}" )

    
    df = pd.DataFrame({'date': dates, 'equity': equityCurve})

    df.to_csv('testing.csv', index=False)


    print(allocatedSharesTotal/len(signals))
    print(dfs.keys())
    df_allocations = pd.DataFrame((allocatedSharesTotal/len(signals)).reshape(1, -1), columns=dfs.keys())
    df_allocations.to_csv("allocationShares.csv")

    plt.plot(dates, equityCurve)
    plt.plot(dates, equityCurve_no_fee)

    #plt.plot(df.index, df['Open'])



    plt.show()


    




        


    

    #return signals



'''def allocateFromFile(filename='signals.csv', minSignal=0.001):
    df_signals = pd.read_csv(filename)

    df_prices = pd.read_csv("2015-2025.csv")
    df_prices = df_prices.set_index('Date')
    endDate = df_prices.index.values[-1]
    df_close = df_prices[[col for col in df_prices.columns if "Close" in col]]
    df_open = df_prices[[col for col in df_prices.columns if "Open" in col]]
    df_close = df_close.rename(columns={ col.split("_")[-1] for col in df_close.columns})
    df_open = df_open.rename(columns={ col.split("_")[-1] for col in df_open.columns})
    df_returns = (df_close - df_open)/df_open




    df_returns_aligned = df_returns.loc[df_allocation.index]


    df_prices = df_prices.rename_axis()

    daysToPredict = 365*7
    dates = pd.date_range(end=endDate, periods=daysToPredict, freq='B')
    
    df_signals.index = dates

    def keep_n_largest(lst, n):
        # Get the indices of the n largest non-zero elements
        non_zero_indices = [i for i, val in enumerate(lst) if val > 0]
        if len(non_zero_indices) <= n:
            return lst.copy()

        # Find the n largest values and their indices
        top_n_indices = sorted(non_zero_indices, key=lambda i: lst[i], reverse=True)[:n]

        # Build new list, zeroing out the rest
        return [val if i in top_n_indices else 0 for i, val in enumerate(lst)]
    



    # Perform allocations
    lst_relative_allocations = []
    df_relative_allocations = df_signals.copy()
    for col in df_relative_allocations.columns:
        df_relative_allocations[col] = df_relative_allocations[col].apply(lambda x: x if x > minSignal else -1000)

    

    for date, row in df_signals.iterrows():




    for i in range(len(dates)):
        signals_day = []
        for key, value in df_signals.items():
            signals_day.append(value[i])
        signals_day = [i if i > 0.001 else 0 for i in signals_day]
        #signals_day = [i if i > -0.0000 else -1000 for i in signals_day]
        signals_day = keep_n_largest(signals_day, 5) # Maximum 5 trades per day to limit commissions
        #signals_day = [i if i > 0.000 else 0 for i in signals_day]
        if np.sum(signals_day) > 0:
            relativeAllocation = np.array(signals_day)/np.sum(signals_day)
            assert(np.abs(np.sum(relativeAllocation)-1)<10**(-9))
            lst_relative_allocations.append(relativeAllocation)
        else:
            lst_relative_allocations.append(None)
'''


    

if __name__ == "__main__":
    output = generateSignal(365*1, useYFinance=False, writeSignals=False, readSignals=False)
    #print(output)