# TODO: Mar 3, 21

## Overall:
* Better documentaion, both within the read me and within the main doc as well. Make code more legible and more generalized.
* Handle nearly any user entered date format, for ease of use
* Make more classes to do more things (want to add one for a lot of financial calcluators)
* Long term, I'd also like to implement machine learning algorithms for stock prediction

## Stock class:
* create binomial tree prediction?
* Financial statement analysis

### __init__
* Handle different frequencies of trade data

### fastGBMpredict
* Ensure new methods of simulation work
* Add ability to use GARCH with fastGBMpredict
* Add ability to pass user defined probability distribution for W<sub>t</sub>
* Add ability to pass user defined drift and/or sigma function/calculation 
* Allow user to mess with arch_model parameters

### mktbeta
* Pretty sure you can get this from yfinance as well so I'd like to give that option 

### optionprice
* Allow user to pass in parameters for fastGBMpredict
* Add a method that uses binomial tree
* Add different ways to calculate risk free rate

## two_stocks class
* this is gone next commit I think, I had something specific in mind but I can no longer remember

## several_stocks
* Financial statement analysis
* Once the above is added, Fama French factor sorts, panelOLS 
* Add different way to calculate covariance matrix (thinking about systemic etc)
* Compare created portfolio historical returns to benchmark of choosing
* Compare created portfolio projected returns to projected returns of benchmark
* Allow user to pass in weights with ticker data to create their own portfolio (this may be created as another class)
* Function to use optimization to find best portfolio weights

### __init__
* Be able to use different frequencies
* Clean the code here, feels VERY messy

### exp_rets
* Make sure this isn't needlessly slow

### min_var_port 
* Allow expected returns to be passed in, then prediction parameters can be set
* Allow prediction parameters to be passed by user in min_var_port
* Add in tangency portfolio

## cashflow_tl class
* Create more shortcuts like addbond
* Find present value of all cashflows (this ties to the ID problem, I have ideas)
* Create a function that makes a SQL table of cashflow dictionary
* Allow SQL tables or other cashflow dictionaries to be passed in to add more cashflows or do calculations

### addbond
* Have an id tag associated with the bond whenever the function is called

### to_dataframe
* Be able to pass in a "groupby" (i.e. all bond cashflows will be combined, all annuity cashflows will be combined, etc)
* Present ID of each cashflow column in the dataframe
