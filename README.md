# f-package
Objects so far: Stock, two_stocks, several_stocks, cashflow_tl
---> todo list/plans found in that doc

----------Stock----------:

-----__init__
Stock object takes ticker in as string, start and end dates in form of 'YYYY-MM-DD' and downloads the data using the yfinance package. 
It also sets self.ticker equal to a yfinance ticker object so you can do anything you want with that when you've created the object.

-----fastGBMpredict:
Uses geometric Brownian motion to create a user-defined number of paths for a user-defined number of days. Mu and Sigma are calculated
from the downloaded data by default. sig2=='ARCH' uses arch_model to find those regression coefficients. They are then used to 
recalculate sigma day by day. mu=='mktdriftmean' finds the stock beta to S and P 500 and multiplies it by the average return in 
the same time period of the downloaded data. The function returns the average of the final day of all the price_paths. The
price paths are accessible using self.price_paths for plotting purposes

-----mktbeta:
Uses statsmodel ols to regress the log returns of the chosen ticker against the market portfolio of your choice to determine beta.

-----option_price:
Takes in strike and expiry (by date or days) and then runs fastGBMpredict to estimate the price of an option

----------two_stocks----------:

__init__
Does basically the same as stock, but with two tickers. Will likely remove this.

----------several_stocks----------:

-----__init__
Takes in several tickers in a list, like stock, but keeps them all together for ease.

-----fastGBMpredict
See above

-----exp_rets
Uses fastGBMpredict to find the expected returns of each stock downloaded

-----min_var_port
Calculates minimum variance portfolio and 'some other portfolio' (i forget the technical term) for the given stocks. returns the weights for the MVP
but also stores the other weights self.w_naught_port and so on. If a target return is passed the function will also find the weights of the allocated 
portfolio (some linear combination of w_0 and w_1)


----------cashflow_tl----------

-----__init__
accepts a start date and date format. Date format doesn't do anything as of yet

-----addBond
Takes coupon rate(annual, in decimal), face value, ytm(useless for now), compounding per annum, and time to maturity in years. It then adds the cashflows
to self.cashflows (dictionary where key = date, value = list of flows on that date) using addcashflows().

-----addcashflows
Takes in a list of cashflows with format [datetime.datetime(YYYY, M, D), AMT] and adds them to self.cashflows. If multiple cashflows have the same date
they are listed beneath the same date key value in a list. 

-----to_dataframe
Creates a pandas dataframe using self.cashflows



