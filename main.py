import numpy as np
import pandas as pd
import yfinance as yf
import re
import datetime
import statsmodels.api as smf
from itertools import accumulate
import matplotlib.pyplot as plt
from arch import arch_model
from numba import jit, vectorize, float64

class Stock():
    def __init__(self, ticker, start_date, end_date, log_rets_dat='Adj Close'):
        '''

        Parameters
        ----------
        ticker : String
            String of ticker of stock you want to download, has to be on 
            Yahoo finance
        start_date : String
              Format 'YYYY-MM-DD'
        end_date : String
            Format 'YYYY-MM-DD'
        log_rets_dat : String, optional
            Data used to find log rets. The default is 'Adj Close'.

        Returns
        -------
        None.

        '''
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
        '''
        Uses Geometric Brownian Motion to Monto Carlo simulate and generate
        stock data

        Parameters
        ----------
        forecast_length : Int
            Number of trading days to calculate in the future.
        num_paths : Int
            Number of price paths you want to make
        mu : Float, optional
            Number to use as drift. The default is the average of the data.
        sig2 : Float, optional
            Method or value used for volatility. The default is volatility of 
            downloaded data.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        Float
            Average of final day of every price path.

        '''
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
        elif sig2=='GARCH':
            am = arch_model(self.log_rets*100, vol = 'Garch', p = 1, o = 0, dist = 'Normal' )
            res = am.fit(disp='off')
            omega, alpha, beta = res.params[1:]
            omega, alpha, beta = omega/10000, alpha/100, beta/100
            drift = mu
            vol = np.zeros((num_paths, forecast_length))
            rets = np.zeros((num_paths, forecast_length))
            
            def g(rets, vol, forecast_length, num_paths):
                for j in range(num_paths):
                    for i in range(1, forecast_length):
                        vol[j, i] = np.sqrt(omega + alpha*rets[j, i-1]**2 + beta*vol[j, i-1]**2)
                        rets[j, i] = drift - 0.5*vol[j, i]**2 + vol[j, i]*np.random.normal()
                return vol, rets
            
            vol, rets = g(rets, vol, forecast_length, num_paths)
            self.price_paths = start_price*np.exp(np.cumsum(rets, axis=1))
            return np.mean(self.price_paths[:, -1])
            
            
        nums = np.random.normal(size=(num_paths, forecast_length-1))
        paths = np.zeros((num_paths, forecast_length))
        paths[:, 1:] = mu - 0.5*sig2 + np.sqrt(sig2)*nums
        paths = np.add.accumulate(paths, axis=1)
        self.price_paths = start_price*np.exp(paths)
        return np.mean(self.price_paths[:, -1])
    
    
    def mktbeta(self, mktport='^GSPC', stock_data='Adj Close'):
        '''
        Finds correlation with a market portfolio

        Parameters
        ----------
        mktport : String, optional
            Ticker of stock/index you want to use as 
            the market. The default is '^GSPC'.
        stock_data : TYPE, optional
            DESCRIPTION. The default is 'Adj Close'.

        Returns
        -------
        Array-like
            Beta and intercept from regression.

        '''
        Y = self.log_rets
        mkt = yf.download( mktport, self.start_date, self.end_date )
        self.mktlogrets = np.log(mkt[stock_data]/mkt[stock_data].shift(1))[1:]
        X = self.mktlogrets
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
        

    
    def fastGBMpredict(self, forecast_length, num_paths, start_price, mu=None, sig2=None, i=0):
        if mu ==None:
            mu=self.data_array
        elif mu=='mktdriftmean':
            beta = self.mktbeta()[-1]
            mu = beta*np.mean(self.mktlogrets)
        if sig2==None:
            sig2=self.var_array
        elif sig2=='ARCH':
            am = arch_model(self.ln_rets[i]*100, vol = 'Arch', p = 1, o = 0, dist = 'Normal' )
            result = am.fit( update_freq = 5, disp='off' )
            _, gamma, alpha = result.params
            @vectorize([float64(float64, float64)])
            def func(x, y):
                return x + mu - 0.5*(gamma + alpha*x**2)*1e-5 + 1e-2*np.sqrt(gamma + alpha*x**2)*np.random.normal()
            paths = func.accumulate(np.zeros((num_paths, forecast_length)), axis=1)
            self.price_paths = start_price*np.exp(paths)
            return np.mean(self.price_paths[:, -1])
        elif sig2=='GARCH':
            am = arch_model(self.ln_rets[i]*100, vol = 'Garch', p = 1, o = 0, dist = 'Normal' )
            res = am.fit(disp='off')
            omega, alpha, beta = res.params[1:]
            omega, alpha, beta = omega/10000, alpha/100, beta/100
            drift = mu
            vol = np.zeros((num_paths, forecast_length))
            rets = np.zeros((num_paths, forecast_length))
            
            def g(rets, vol, forecast_length, num_paths):
                for j in range(num_paths):
                    for i in range(1, forecast_length):
                        vol[j, i] = np.sqrt(omega + alpha*rets[j, i-1]**2 + beta*vol[j, i-1]**2)
                        rets[j, i] = drift - 0.5*vol[j, i]**2 + vol[j, i]*np.random.normal()
                return vol, rets
            
            vol, rets = g(rets, vol, forecast_length, num_paths)
            self.price_paths = start_price*np.exp(np.cumsum(rets, axis=1))
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
        elif sig2=='GARCH':
            sig2 = ['GARCH' for i in range(len(self.tickers))]
        elif sig2=='ARCH':
            sig2 = ['ARCH' for i in range(len(self.tickers))]
        if start_price == None:
            start_price = np.array([self.frames[i]['Adj Close'][-1] for i in range(len(self.tickers))])
        rets = []
        for i in range(len(self.tickers)):
            rets.append(self.fastGBMpredict(20, 1000, start_price[i], mu[i], sig2[i], i))
        return np.array(rets)
        
    def min_var_port(self, exp_rets = None, Target=None):
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
    
    
class cashflow_tl():
    def __init__(self, start_date, date_format='yyyy-mm-dd'):
        self.start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        self.date_format = date_format
        self.cashflows = {self.start_date:[0]}
        
        
    def addBond(self, coupon_rate, face_val, ytm, comp_per_annum, ttm, **kwargs):
        inter = datetime.timedelta(days=(365/comp_per_annum))
        coupon = (coupon_rate*face_val)/comp_per_annum
        bond = np.array([ [self.start_date + inter*(i+1), coupon] for i in range(ttm*comp_per_annum)])
        bond[-1, -1] += face_val
        self.addcashflows(bond)
        
    def addcashflows(self, list_of_flows):
        for i in range(len(list_of_flows[:,0])):
            if list_of_flows[i,0] in self.cashflows:
                self.cashflows[list_of_flows[i,0]].extend(list_of_flows[i,1:])
            else:
                self.cashflows[list_of_flows[i,0]] = list_of_flows[i,1:].tolist()
        
    def to_dataframe(self):
        self.dfflow = pd.DataFrame.from_dict(self.cashflows, orient='index').fillna(0)
        
class finCalcs:
    def bond_price(face_val, compounding_freq, ytm, TTM, coupon_rate):
        price = 0
        ytm = ytm/compounding_freq
        coupon = (coupon_rate*face_val)/compounding_freq
        n = compounding_freq*TTM
        oneover = 1/(1+ytm)**n
        price = coupon*((1-oneover)/ytm) + face_val*oneover
        return price
    
    def duration(face_val, compounding_freq, ytm, TTM, coupon_rate, type_of='Modified-Macaulay'):
        price = finCalcs.bond_price(face_val, compounding_freq, ytm, TTM, coupon_rate)
        ytm = ytm/compounding_freq
        coupon = (coupon_rate*face_val)/compounding_freq
        n = compounding_freq*TTM
        oneover = 1/(1+ytm)**n
        dpdi = -(coupon/ytm**2)*(1 - oneover) + (coupon*n/ytm)*(1/(1+ytm)**(n+1)) - face_val*n/(1+ytm)**(n+1)
        if type_of == 'Modified-Macaulay':
            return dpdi/price
        elif type_of == 'Macaulay':
            return -(1 + ytm)*dpdi/price
