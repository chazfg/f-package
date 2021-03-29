import numpy as np
import pandas as pd
import yfinance as yf
import re
from datetime import datetime
import statsmodels.api as smf
from itertools import accumulate

class Stock():
    def __init__(self, ticker, start_date, end_date, log_rets_dat='Adj Close'):
        ticker = ticker.upper()
        self.ticker = yf.Ticker(ticker)
        self.price_paths = None
        self.start_date = start_date
        self.end_date = end_date
        yr0, mo0, day0 = list(map(int, start_date.split('-')))
        yr1, mo1, day1 = list(map(int, end_date.split('-')))
        self.current_data = yf.download( ticker, start_date, end_date )
        self.log_rets = np.log(self.current_data[log_rets_dat]/self.current_data[log_rets_dat].shift(1))[1:]
        self.data_var = np.var(self.log_rets)
        self.data_mean = np.mean(self.log_rets)
        
   
    def fastGBMpredict(self, forecast_length, num_paths, mu=None, sig2=None, **kwargs):
        start_price = np.asarray(self.current_data['Adj Close'])[-1]
        if mu ==None:
            mu=self.data_mean
        elif mu=='mktdriftmean':
            beta = self.mktbeta()[-1]
            mu = beta*np.mean(self.mktlogrets)
        if sig2==None:
            sig2=self.data_var
        elif sig2=='ARCH':
            am = arch_model(self.log_rets*100, vol = 'Arch', p = 1, o = 0, dist = 'Normal' )
            result = am.fit( update_freq = 5, disp='off' )
            _, gamma, alpha = result.params
            @vectorize([float64(float64, float64)])
            def func(x, y):
                return x + mu - 0.5*(gamma + alpha*x**2)*1e-5 + 1e-2*np.sqrt(gamma + alpha*x**2)*np.random.normal()
            paths = func.accumulate(np.zeros((num_paths, forecast_length)), axis=1)
            self.price_paths = start_price*np.exp(paths)
            return np.mean(self.price_paths[:, -1])
            
        nums = np.random.normal(size=(num_paths, forecast_length-1))
        paths = np.zeros((num_paths, forecast_length))
        paths[:, 1:] = mu - 0.5*sig2 + np.sqrt(sig2)*nums
        paths = np.add.accumulate(paths, axis=1)
        self.price_paths = start_price*np.exp(paths)
        return np.mean(self.price_paths[:, -1])
    
    def mktbeta(self, mktport='^GSPC', stock_data='Adj Close'):
        Y = self.log_rets
        mkt = yf.download( mktport, self.start_date, self.end_date )
        mktlogrets = np.log(mkt[stock_data]/mkt[stock_data].shift(1))[1:]
        X = mktlogrets
        X = smf.add_constant(X)
        self.regression_data = smf.OLS(Y, X).fit()
        return self.regression_data.params
    
    
    def option_price(self, strike, expiry, start_price=None, r_f=0.06, contract='call'):
        if start_price==None:
            start_price = self.current_data['Adj Close'].iloc[0]
        if isinstance(expiry, int):
            TTM = expiry
        elif re.compile('....-..-..').search(expiry):
            d2 = datetime.strptime(expiry, "%Y-%m-%d")            
            d1 = datetime.strptime(self.end_date, "%Y-%m-%d")
            TTM = int((d2-d1).total_seconds()/86400)
            
        if  self.price_paths == None or self.price_paths.any() == None:
            self.GBMpredict(TTM+1, 1000)
            prices = self.price_paths[:, -1]
        
        if contract == 'call':
            count = prices > strike
            return (np.mean(prices[count]) - strike)*np.exp(-0.06*(TTM/252))
        else:
            count = prices < strike
            return (np.mean(strike - prices[count]))*np.exp(r_f*(TTM/252))
        
class two_stocks():
    def __init__(self, tickers, start_date, end_date):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.data = pd.DataFrame()
        frames = []
        for stock in tickers:
            setattr(self, stock, yf.Ticker(stock))
            frames.append(yf.download(stock, start_date, end_date))
        self.data = pd.concat(frames, keys=tickers)   
    
class several_stocks():
    def __init__(self, tickers, start_date, end_date):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.data = pd.DataFrame()
        self.frames = []
        for stock in tickers:
            setattr(self, stock, yf.Ticker(stock))
            self.frames.append(yf.download(stock, start_date, end_date))
        self.data = pd.concat(self.frames, keys=tickers)
        self.ln_rets = np.zeros((len(tickers), len(self.frames[0]['Adj Close']) - 1))
        for i, frame in enumerate(self.frames):
            self.ln_rets[i] = np.asarray(np.log(frame['Adj Close']/frame['Adj Close'].shift()))[1:]
        self.mean_array = np.array([np.mean(self.ln_rets[i]) for i in range(len(tickers))])
        self.var_array = np.array([np.var(self.ln_rets[i]) for i in range(len(tickers))])
        

    
    def fastGBMpredict(self, forecast_length, num_paths, mu=None, sig2=None, **kwargs):
        start_price = np.asarray(self.current_data['Adj Close'])[-1]
        if mu ==None:
            mu=self.data_mean
        elif mu=='mktdriftmean':
            beta = self.mktbeta()[-1]
            mu = beta*np.mean(self.mktlogrets)
        if sig2==None:
            sig2=self.data_var
        elif sig2=='ARCH':
            am = arch_model(self.log_rets*100, vol = 'Arch', p = 1, o = 0, dist = 'Normal' )
            result = am.fit( update_freq = 5, disp='off' )
            _, gamma, alpha = result.params
            @vectorize([float64(float64, float64)])
            def func(x, y):
                return x + mu - 0.5*(gamma + alpha*x**2)*1e-5 + 1e-2*np.sqrt(gamma + alpha*x**2)*np.random.normal()
            paths = func.accumulate(np.zeros((num_paths, forecast_length)), axis=1)
            self.price_paths = start_price*np.exp(paths)
            return np.mean(self.price_paths[:, -1])
            
        nums = np.random.normal(size=(num_paths, forecast_length-1))
        paths = np.zeros((num_paths, forecast_length))
        paths[:, 1:] = mu - 0.5*sig2 + np.sqrt(sig2)*nums
        paths = np.add.accumulate(paths, axis=1)
        self.price_paths = start_price*np.exp(paths)
        return np.mean(self.price_paths[:, -1])
    
    def exp_rets(self, forecast_length=None, num_paths=None, mu=None, sig2=None, start_price=None):
        if forecast_length == None:
            forecast_length = 20
        if num_paths == None:
            num_paths = 1000
        if mu==None:
            mu = self.mean_array
        if sig2==None:
            sig2 = self.var_array
        if start_price == None:
            start_price = np.array([self.frames[i]['Adj Close'][-1] for i in range(len(self.tickers))])
        return np.array([self.fastGBMpredict(forecast_length, num_paths, sig2[i], mu[i], start_price[i]) for i in range(len(self.tickers))])
        
    def min_var_port(self, Target=None):
        e = np.ones(len(self.tickers))
        r = self.exp_rets()
        cov = np.cov(self.ln_rets)
        inv_cov = np.linalg.inv(cov)
        a = e @ inv_cov @ e.T
        b = r @ inv_cov @ e.T
        c = r @ inv_cov @ r.T
        self.w_naught_port = inv_cov @ e / a
        self.w_one_port = inv_cov @ r / b
        self.w_naught_port_stats = [b/a, 1/a]
        self.w_one_port_stats = [c/b, c/b**2]
        if Target is not None:
            psi = (a*b*Target - b**2)/(a*c - b**2)
            self.w_alloc = (1-psi)*self.w_naught_port + psi*self.w_one_port
        return inv_cov @ e / a
    
