import yfinance as yf
import pandas as pd
import os


folder_path = './Data/data_not_bias'

file_names = [filename for filename in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, filename))]
file_names = file_names
tickers = [i[:-4] for i in file_names]

print(tickers)

data = yf.download(tickers, start='2015-01-01', end='2025-01-01')
#data = yf.download(tickers, start='2024-01-01', end='2025-01-01')

data = data.reset_index(['Date'])

print(data.columns)

data = data[[(  'Date',     '')] + [ ( 'Close', ticker) for ticker in tickers] + [ ( 'Open', ticker) for ticker in tickers] + [ ( 'Volume', ticker) for ticker in tickers] + [ ( 'High', ticker) for ticker in tickers] + [ ( 'Low', ticker) for ticker in tickers] ] #+  ,  [ ( 'Volume', ticker) for ticker in tickers]]
#data = data.rename(columns={( 'Date','') : ( 'Date','Date')})
#data = data.rename(columns={  ( 'Close', ticker) :  ( 'Close', ticker+'_Close') for ticker in tickers})
#data = data.rename(columns={  ( 'Open', ticker) :  ( 'Open', ticker+'_Open') for ticker in tickers})
#data = data.rename(columns={  ( 'Volume', ticker) :  ( 'Volume', ticker+'_Volume') for ticker in tickers})

data.columns = data.columns.map(lambda x: '_'.join([str(i) for i in x]))


#data.columns = data.columns.droplevel()
data = data.rename(columns={'Date_' : 'Date'})


print(data.columns)


print(data.head(5))


data.to_csv("2015-2025.csv")