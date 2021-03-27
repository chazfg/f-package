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
        
    def GBMpredict(self, forecast_length, num_paths, sig2=None, mu=None):
        if mu==None:
            mu=self.data_mean
        if sig2==None:
            sig2=self.data_var
        start_price = np.asarray(self.current_data['Adj Close'])[-1]
        paths = np.zeros((num_paths, forecast_length))
        for i in range(num_paths):
            for j in range(1, forecast_length):
                paths[i, j] = paths[i, j-1] + (mu - 0.5*sig2) + np.sqrt(sig2)*np.random.normal()
                
                    
        self.price_paths = start_price*np.exp(paths)
        return np.mean(self.price_paths[:,-1])
    
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
        dl = ""
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
        dl = ""
        frames = []
        for stock in tickers:
            setattr(self, stock, yf.Ticker(stock))
            frames.append(yf.download(stock, start_date, end_date))
        self.data = pd.concat(frames, keys=tickers)
