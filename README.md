# Worskshop-2
Workshop 2 codes and projects

**Options Trading Analysis**

This script is designed for analyzing stock options using the Black-Scholes-Merton (BSM) formula and real market data. The script is written in Python and utilizes various libraries for financial data retrieval, calculation, and analysis.

**Libraries Used:**
pandas: For handling and manipulating data in tabular form.

numpy: For numerical operations and calculations.

yfinance: For fetching stock-related data from Yahoo Finance.

scipy.stats: For statistical operations and calculations.

datetime: For handling date and time data.

openpyxl: For reading and writing Excel files.

**Functions:**
- get_stock_price(ticker: str = 'AAPL')
  
Fetches the current stock price of the given ticker from Yahoo Finance.

- stock_volatility(ticker: str = 'AAPL', start_date: str = '2022-01-01', end_date: str = '2023-01-01')
  
Calculates the stock volatility using historical data from Yahoo Finance.

- get_riskfree_rate()
  
Retrieves the risk-free rate using the 10-year US Treasury Note yield from Yahoo Finance.

- get_dividend_yield(ticker: str = 'AAPL')
  
Fetches the dividend yield of the given stock from Yahoo Finance.

- call_option_pricing_bsm(S0: float, X: float, T: float, vol: float, rfr: float, dy: float)
  
Calculates the price of a call option using the Black-Scholes-Merton formula.

- put_option_pricing_bsm(S0: float, X: float, T: float, vol: float, rfr: float, dy: float)
  
Calculates the price of a put option using the Black-Scholes-Merton formula.

- min_max_option(S0: float, X: float, T: float, rfr: float, op_type: str)
  
Calculates the minimum and maximum price for a given option using non-arbitrage bounds.

- get_option_call_strike_prices(ticker: str = 'AAPL')
  
Retrieves call option strike prices and expiration dates for a given stock.

- get_option_put_strike_prices(ticker: str = 'AAPL')
  
Retrieves put option strike prices and expiration dates for a given stock.

- strategies_options(S0, X, T, rfr: float, op_type: str, op_price)
  
Analyzes various strategies for each option available based on market prices and non-arbitrage bounds.

- bsm_pricing_option_array(S0: float, X, T, vol: float, rfr: float, dy: float, op_type: str)
  
Calculates BSM prices for an array of options.

- compare_bsm_real_price(bsm_prices, real_prices)
  
Compares BSM prices with real market prices and returns conclusions.

- user_experience()
  
Provides an interactive user experience to explore stock option information, calculate option prices, and analyze available options.

**How to Use:**

1. Ensure you have the required Python libraries installed. You can install them using:

 pip install pandas numpy yfinance scipy openpyxl

2. Copy and paste the entire script into a Python environment or script.

3. Run the script.

4. The script will interactively guide you through different options, allowing you to:

- Retrieve stock information (price, volatility, dividend yield, risk-free rate).
  
- Calculate minimum and maximum option prices.
  
- Calculate the price of a specific option.
  
- View all available options, analyze strategies, and compare BSM prices with real market prices.

5. The script will save the results into Excel files for further analysis.
