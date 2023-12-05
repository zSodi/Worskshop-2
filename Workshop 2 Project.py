import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import norm
from datetime import datetime
import openpyxl


# Function to get a stock price from Yahoo Finance
def get_stock_price(ticker: str = 'AAPL'):
    """
    Get the actual stock price from Yahoo Finance
    :param ticker: Stock ticker (string)
    :return: actual stock price (float)
    """
    try:
        # Instructions for stock price
        stock_data = yf.Ticker(ticker)
        stock_price = stock_data.info['ask']
        return stock_price
    # Error print in case Yahoo Finance doesn't have the data
    except Exception as e:
        print(f'Unable to know the stock price {ticker}: {e}')
        return None


# Function to get stock volatility from Yahoo Finance
def stock_volatility(ticker: str = 'AAPL', start_date: str = '2022-01-01', end_date: str = '2023-01-01'):
    """
    Calculate stock volatility using Yahoo Finance
    :param ticker: Stock ticker (string)
    :param start_date: Data start date (string -> year/month/day)
    :param end_date_str: Data end date (string -> year/month/day)
    :return: stock volatility (float)
    """

    # Get data
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    # Daily returns
    stock_data['Daily Return'] = stock_data['Adj Close'].pct_change()
    # Std deviation (volatility)
    volatility = stock_data['Daily Return'].std()
    return volatility


# Function to get the risk-free rate
def get_riskfree_rate():
    """
    Gets the risk-free rate using 10 Year US Treasury Note yield from Yahoo Finance
    :return: risk-free rate (float)
    """
    # Get historical data
    risk_free_data = yf.download(tickers='^IRX', period='1d')
    # Most recent yield convert it to decimal
    riskfree_rate = risk_free_data['Close'].iloc[-1] / 100
    return riskfree_rate


# Function to get a stock dividend yield
def get_dividend_yield(ticker: str = 'AAPL'):
    """
    Get a stock dividend yield from Yahoo Finance
    :param ticker: Stock ticker (string)
    :return: Stock dividend yield (float)
    """
    try:
        # Instructions for dividend yield
        stock_data = yf.Ticker(ticker)
        dividend_yield = stock_data.info['dividendYield']
        return dividend_yield
    # Error print in case Yahoo Finance doesn't have the data
    except Exception as e:
        print(f'Unable to know dividen yield from {ticker}: {e}')


# Function to price equity call options with Black-Scholes formula
def call_option_pricing_bsm(S0: float, X: float, T: float, vol: float, rfr: float, dy: float):
    """
    Calculate a call option price using BSM formula -> c= S0*N(d1) - X*N(d2)*e^-rT
    :param S0: Stock price (float)
    :param X: Stock strike price (float)
    :param T: Period (in years) (float)
    :param vol: Stock volatility (float)
    :param rfr: Risk-free rate (float)
    :param dy: Stock dividend yield (float)
    :return: Call option price (float)
    """
    # Calculate the probability if stock price > stock strike price
    d1 = (np.log(S0 / X) + (rfr - dy + vol ** 2 / 2) * T) / (vol * np.sqrt(T))
    # Calculate the probability if stock price > stock strike price if the stock price was above before the strike
    d2 = d1 - vol * np.sqrt(T)
    # Use normal distribution
    N = norm(0, 1).cdf
    # Calculate the formula
    call_price = (S0 * np.exp(-dy * T) * N(d1)) - (X * np.exp(-rfr * T) * N(d2))
    return call_price


# Function to price equity put options with Black-Scholes formula
def put_option_pricing_bsm(S0: float, X: float, T: float, vol: float, rfr: float, dy: float):
    """
    Calculate a put option price using BSM formula -> p = X*e^-rT*N(-d2) - S0*N(-d1)
    :param S0: Stock price (float)
    :param X: Stock strike price (float)
    :param T: Period (in years) (float)
    :param vol: Stock volatility (float)
    :param rfr: Risk-free rate (float)
    :param dy: Stock dividend yield (float)
    :return: Call option price (float)
    """
    # Calculate the probability if stock price > stock strike price
    d1 = (np.log(S0 / X) + (rfr - dy + vol ** 2 / 2) * T) / (vol * np.sqrt(T))
    # Calculate the probability if stock price > stock strike price if the stock price was above before the strike
    d2 = d1 - vol * np.sqrt(T)
    # Use normal distribution
    N = norm(0, 1).cdf
    # Calculate the formula
    put_price = (X * np.exp(-rfr * T) * N(1 - d2)) - (S0 * np.exp(-dy * T) * N(1 - d1))
    return put_price


# Function to calculate min and max of an option price
def min_max_option(S0: float, X: float, T: float, rfr: float, op_type: str):
    """
    Calculates the minimum and maximum price for an option
    :param S0: Stock price (float)
    :param X: Stock strike price (float)
    :param T: Period (years) (float)
    :param rfr: Risk-Free Rate (float)
    :param op_type: Call or Put (string)
    :return: Minimum and maximum option price (array)
    """
    # Conditional if the first letter is a c or a p (call or put)
    if op_type.lower()[0] == 'c':
        # c <= S0
        max = S0
        # c >= S0 - X * e^-rt
        min = S0 - X * np.exp(-rfr * T)
    elif op_type.lower()[0] == 'p':
        # p <= X * e^-rt
        max = X * np.exp(-rfr * T)
        # p >= X * e^-rt - S0
        min = X * np.exp(-rfr * T) - S0
    else:
        print('Wrong option type, try call or put only')
        return None
    return np.array([min, max])


# Function for getting stock option strike prices and the expiration dates
def get_option_call_strike_prices(ticker: str = 'AAPL'):
    """
    Get call option strike prices for a given stock from Yahoo Finance
    :param ticker: Stock ticker
    :return: Tuple containing lists of call and put option strike prices
    """
    try:
        # Get option chain data
        stock_data = yf.Ticker(ticker)
        option_chain = stock_data.options

        # Get strike prices for call and put options
        call_option_data = stock_data.option_chain(option_chain[0])
        # Convert to list
        call_strike_prices = call_option_data.calls['strike'].tolist()
        call_price = call_option_data.calls['lastPrice'].tolist()
        expiration_dates = np.array(list(option_chain)).tolist()
        # Convert list of strings to list of datetime objects
        expiration_dates = [datetime.strptime(date, '%Y-%m-%d') for date in expiration_dates]
        # Get call expiration periods (in days)
        today = datetime.today()
        call_expiration_periods = [(expiration - today).days for expiration in expiration_dates]
        # Do a DataFrame
        call_df = pd.DataFrame(list(zip(call_strike_prices, call_expiration_periods, call_price)),
                               columns=['Strike', 'Expiration', 'Last Price'])
        return call_df

    except Exception as e:
        print(f'Unable to get option strike prices for {ticker}: {e}')


# Function for getting stock option strike prices and the expiration dates
def get_option_put_strike_prices(ticker: str = 'AAPL'):
    """
    Get put option strike prices for a given stock from Yahoo Finance
    :param ticker: Stock ticker
    :return: Tuple containing lists of call and put option strike prices
    """
    try:
        # Get option chain data
        stock_data = yf.Ticker(ticker)
        option_chain = stock_data.options

        # Get strike prices for call and put options
        put_option_data = stock_data.option_chain(option_chain[0])
        # Convert to list
        put_strike_prices = put_option_data.puts['strike'].tolist()
        put_price = put_option_data.puts['lastPrice'].tolist()
        expiration_dates = np.array(list(option_chain)).tolist()
        # Convert list of strings to list of datetime objects
        expiration_dates = [datetime.strptime(date, '%Y-%m-%d') for date in expiration_dates]
        # Get call expiration periods (in days)
        today = datetime.today()
        put_expiration_periods = [(expiration - today).days for expiration in expiration_dates]
        # Do a DataFrame
        puts_df = pd.DataFrame(list(zip(put_strike_prices, put_expiration_periods, put_price)),
                               columns=['Strike', 'Expiration', 'Last Price'])
        return puts_df

    except Exception as e:
        print(f'Unable to get option strike prices for {ticker}: {e}')


# Function for iterate over all the options available and know each strategy
def strategies_options(S0, X, T, rfr: float, op_type: str, op_price):
    """
    Do a DataFrame for know a strategy for each option available
    :param stock_price: Stock Price
    :param X: Strike Price
    :param period/360: Period (years) (float)
    :param rfr: Risk-Free Rate (float)
    :param op_type: Call or Put (string)
    :param op_price: Option Price
    :return:
    """
    # Empty observations list
    strat_list = []
    min_arb_list = []
    for i in range(len(op_price)):
        option_price = op_price[i]
        strike_price = X[i]
        period = T[i]
        # Option bounds
        op_bounds = min_max_option(S0, strike_price, period / 360, rfr, op_type)
        if op_bounds is None:
            return None
        # Out of bounds conditions
        isLBOff = option_price < op_bounds[0]
        isUBOff = option_price > op_bounds[1]
        isOOB = isLBOff or isUBOff
        # If out of bounds, check side
        if isOOB:
            if op_type.lower()[0] == 'c':
                if isLBOff:  # LB off
                    strat_list.append(
                        f'Short Stock at {S0:,.2f} + Buy Call at {option_price:,.3f}. Invest {S0 - option_price:,.3f} at {rfr:,.2%} for {period / 360:,.2f} years')
                    future_invest = (S0 - option_price) * np.exp(rfr * period / 360)
                    min_arb_amt = future_invest - strike_price
                    min_arb_list.append(min_arb_amt)
                else:  # UB off
                    strat_list.append(f'Sell Call at {option_price:,.3f} + Buy Stock at {S0:,.2f}')
                    min_arb_amt = option_price - S0
                    min_arb_list.append(min_arb_amt)
            elif op_type.lower()[0] == 'p':
                if isLBOff:  # LB off
                    strat_list.append(
                        f'Buy Stock + Put. Borrow {S0 + option_price:,.3f} at {rfr:,.2%} for {period / 360:,.2f} years\n')
                    future_liab = -1 * (S0 + option_price) * np.exp(rfr * period / 360)
                    min_arb_amt = future_liab + strike_price
                    min_arb_list.append(min_arb_amt)
                else:  # UB off
                    strat_list.append(
                        f'Sell Put at {option_price:,.3f}. Invest premium at {rfr:,.2%} for {period / 360:,.2f} years')
                    min_arb_amt = option_price * np.exp(rfr * period / 360) - strike_price
                    min_arb_list.append(min_arb_amt)
            else:
                print('Incorrect option type!')
        else:
            min_arb_list.append('Option price is between non-arbitrage bounds.')
            strat_list.append('Option price is between non-arbitrage bounds.')
    return min_arb_list, strat_list


# Functions to compare options pricing using BSM and the real market prices
def bsm_pricing_option_array(S0: float, X, T, vol: float, rfr: float, dy: float, op_type: str):
    """
    Calculate call price option for an array
    :param S0: Stock price (float)
    :param X: Strike Price (array float)
    :param T: Periods (array float)
    :param vol: Stock volatility (float)
    :param rfr: Risk-Free Rate (float)
    :param dy: Stock dividend yield (float)
    :param op_type: Call or Put (string)
    :return: BSM call option price
    """
    bsm_prices = []
    # Loop through the arrays and get each BSM Price
    for i in range(len(X)):
        # Check if it is a call or option request
        if op_type.lower()[0] == 'c':
            bsm_price = call_option_pricing_bsm(S0=S0, X=X[i], T=T[i], vol=vol, rfr=rfr, dy=dy)
            bsm_prices.append(bsm_price)
        elif op_type.lower()[0] == 'p':
            bsm_price = put_option_pricing_bsm(S0=S0, X=X[i], T=T[i], vol=vol, rfr=rfr, dy=dy)
            bsm_prices.append(bsm_price)
    return bsm_prices


# Function to check if the real price is greater or less than BSM Price
def compare_bsm_real_price(bsm_prices, real_prices):
    """
    Compares a BSM Pricing vs the Real Price of an Option
    :param bsm_prices: BSM prices (array floats)
    :param real_prices: Real prices in market (array floats)
    :return: string list with the conclusion if its greater or less
    """
    conclusions = []
    for i in range(len(bsm_prices)):
        if bsm_prices[i] > real_prices[i]:
            conclude = 'BSM Price greater-than Real Price'
            conclusions.append(conclude)
        elif bsm_prices[i] < real_prices[i]:
            conclude = 'BSM Price less-than Real Price'
            conclusions.append(conclude)
        elif bsm_prices[i] == real_prices[i]:
            conclude = 'BSM Pice same as Real Price'
    return conclusions


# Example using AAPL to calculate a price call option

# AAPL Data
stock_ticker = 'AAPL'
aapl_price = get_stock_price()
aapl_volatility = stock_volatility()
aapl_dividend_yield = get_dividend_yield()
risk_free_rate = get_riskfree_rate()

# AAPL Strike Options
aapl_strike_call = 189
aapl_strike_put = 193

# Info print
print(
    f'For {stock_ticker}: \n Price: {aapl_price} \n Volatility: {aapl_volatility} \n Dividend Yield {aapl_dividend_yield} \n '
    f'Risk-Free Rate: {risk_free_rate} \n Strike Price: {aapl_strike_call}')

# Min and Max price for a call or put for AAPL
call_min_max = min_max_option(S0=aapl_price, X=aapl_strike_call, T=0.5, rfr=risk_free_rate, op_type='call')
put_min_max = min_max_option(S0=aapl_price, X=aapl_strike_put, T=0.5, rfr=risk_free_rate, op_type='put')
print(f'A call option for {stock_ticker} the minimum and maximum should be: {call_min_max}')
print(f'A put option for {stock_ticker} the minimum and maximum should be: {put_min_max}')

# Calculate call and put option pricing for AAPL
aapl_call = call_option_pricing_bsm(S0=aapl_price, X=aapl_strike_call, T=0.5, vol=aapl_volatility, rfr=risk_free_rate,
                                    dy=risk_free_rate)
print(f'A call option pricing for {stock_ticker} should be: {aapl_call}')
aapl_put = put_option_pricing_bsm(S0=aapl_price, X=aapl_strike_put, T=0.5, vol=aapl_volatility, rfr=risk_free_rate,
                                  dy=aapl_dividend_yield)
print(f'A put option pricing for {stock_ticker} should be: {aapl_put}')

# Instructions for call the function of strategies
calls_df = get_option_call_strike_prices()
puts_df = get_option_put_strike_prices()
# Call the function for puts and calls
result_calls = strategies_options(S0=aapl_price, X=calls_df['Strike'], T=calls_df['Expiration'], rfr=risk_free_rate,
                                  op_type='call', op_price=calls_df['Last Price'])
result_puts = strategies_options(S0=aapl_price, X=puts_df['Strike'], T=puts_df['Expiration'], rfr=risk_free_rate,
                                 op_type='put', op_price=puts_df['Last Price'])
min_arb_calls, strat_calls = result_calls
min_arb_puts, strat_puts = result_puts

# Create new DF Columns with the results in calls
calls_df['Minimum arbitrage bounds'] = min_arb_calls
calls_df['Strategy for the option'] = strat_calls
print(calls_df)

# Create new DF Columns with the results in puts
puts_df['Minimum arbitrage bounds'] = min_arb_puts
puts_df['Strategy for the option'] = strat_puts

# Call function to get the BSM Prices for the call options
bsm_result_call = bsm_pricing_option_array(S0=aapl_price, X=calls_df['Strike'], T=calls_df['Expiration'],
                                           vol=aapl_volatility, rfr=risk_free_rate, dy=aapl_dividend_yield,
                                           op_type='call')
# Add BSM Price Column
calls_df['BSM Price'] = bsm_result_call

# Call function to get the BSM Prices for the put options
bsm_result_put = bsm_pricing_option_array(S0=aapl_price, X=puts_df['Strike'], T=puts_df['Expiration'],
                                          vol=aapl_volatility, rfr=risk_free_rate, dy=aapl_dividend_yield,
                                          op_type='put')
# Add BSM Price Column
puts_df['BSM Price'] = bsm_result_put

# Call the function of the comparing and add the conclusions to the DataFrame
compare_calls = compare_bsm_real_price(calls_df['BSM Price'], calls_df['Last Price'])
compare_puts = compare_bsm_real_price(puts_df['BSM Price'], puts_df['Last Price'])
calls_df['BSM Price vs Real Price'] = compare_calls
puts_df['BSM Price vs Real Price'] = compare_puts
print(calls_df)
print(puts_df)


# USER EXPERIENCE

# Function so the user can type the tickers he wants
def user_experience():
    user = True
    while user:
        ticker = input('Type a Yahoo Finance ticker to know the Options available and it is information\n')
        price = get_stock_price(ticker)
        volatility = stock_volatility(ticker)
        dividend_yield = get_dividend_yield(ticker)
        risk_free_rate = get_riskfree_rate()
        choice = input('What do you want to know?\n'
                       'a) Information about your ticker (price, volatility, dividend yield, risk-free rate\n'
                       'b) Minimum and maximum for an option price\n'
                       'c) Calculate the price of an option\n'
                       'd) View all options available, strategies and compare prices\n'
                       'e) Exit\n'
                       'Type a, b, c, d or e\n')
        if choice.lower() == 'a':
            # Info print
            print(
                f'For {ticker}: \n Price: {price} \n Volatility: {volatility} \n Dividend Yield {dividend_yield} \n '
                f'Risk-Free Rate: {risk_free_rate}')
            user = False
        elif choice.lower() == 'b':
            choice_option = input('Is your option a call or put?')
            if choice_option.lower()[0] == 'c':
                call_strike = input('Which is your Price Strike?')
                call_period = input('How many days will it has the call option?')
                # Min and Max price for a call or put for AAPL
                call_min_max = min_max_option(S0=price, X=float(call_strike), T=float(call_period) / 360,
                                              rfr=risk_free_rate, op_type='call')
                print(f'A call option for {ticker} the minimum and maximum should be: {call_min_max}')
                user = False
            elif choice_option.lower()[0] == 'p':
                put_strike = input('Which is your Price Strike?')
                put_period = input('How many days will it has the call option?')
                put_min_max = min_max_option(S0=price, X=float(put_strike), T=float(put_period) / 360,
                                             rfr=risk_free_rate, op_type='put')
                print(f'A put option for {ticker} the minimum and maximum should be: {put_min_max}')
                user = False
            else:
                print('Wrong input, please type call or put. You will exit')
                user = False
        elif choice.lower()[0] == 'c':
            choice_option = input('Is your option a call or put?')
            if choice_option.lower()[0] == 'c':
                call_strike = input('Which is your Price Strike?')
                call_period = input('How many days will it has the call option?')
                call = call_option_pricing_bsm(S0=price, X=float(call_strike), T=float(call_period)/360,
                                               vol=volatility,
                                               rfr=risk_free_rate,
                                               dy=dividend_yield)
                print(f'A call option pricing for {ticker} should be: {call}')
                user = False
            elif choice_option.lower()[0] == 'p':
                put_strike = input('Which is your Price Strike?')
                put_period = input('How many days will it has the call option?')
                put = put_option_pricing_bsm(S0=price, X=float(put_strike), T=float(put_period)/360, vol=aapl_volatility,
                                             rfr=risk_free_rate,
                                             dy=dividend_yield)
                print(f'A put option pricing for {ticker} should be: {put}')
                user = False
            else:
                print('Wrong input, please type call or put. You will exit')
                user = False
        elif choice.lower()[0] == 'd':
            # Instructions for call the function of strategies
            calls_df = get_option_call_strike_prices(ticker)
            puts_df = get_option_put_strike_prices(ticker)
            # Call the function for puts and calls
            result_calls = strategies_options(S0=price, X=calls_df['Strike'], T=calls_df['Expiration'],
                                              rfr=risk_free_rate,
                                              op_type='call', op_price=calls_df['Last Price'])
            result_puts = strategies_options(S0=price, X=puts_df['Strike'], T=puts_df['Expiration'],
                                             rfr=risk_free_rate,
                                             op_type='put', op_price=puts_df['Last Price'])
            min_arb_calls, strat_calls = result_calls
            min_arb_puts, strat_puts = result_puts

            # Create new DF Columns with the results in calls
            calls_df['Minimum arbitrage bounds'] = min_arb_calls
            calls_df['Strategy for the option'] = strat_calls

            # Create new DF Columns with the results in puts
            puts_df['Minimum arbitrage bounds'] = min_arb_puts
            puts_df['Strategy for the option'] = strat_puts

            # Call function to get the BSM Prices for the call options
            bsm_result_call = bsm_pricing_option_array(S0=price, X=calls_df['Strike'], T=calls_df['Expiration'],
                                                       vol=volatility, rfr=risk_free_rate, dy=dividend_yield,
                                                       op_type='call')
            # Add BSM Price Column
            calls_df['BSM Price'] = bsm_result_call

            # Call function to get the BSM Prices for the put options
            bsm_result_put = bsm_pricing_option_array(S0=price, X=puts_df['Strike'], T=puts_df['Expiration'],
                                                      vol=volatility, rfr=risk_free_rate, dy=dividend_yield,
                                                      op_type='put')
            # Add BSM Price Column
            puts_df['BSM Price'] = bsm_result_put

            # Call the function of the comparing and add the conclusions to the DataFrame
            compare_calls = compare_bsm_real_price(calls_df['BSM Price'], calls_df['Last Price'])
            compare_puts = compare_bsm_real_price(puts_df['BSM Price'], puts_df['Last Price'])
            calls_df['BSM Price vs Real Price'] = compare_calls
            puts_df['BSM Price vs Real Price'] = compare_puts
            print(calls_df)
            print(puts_df)
            calls_df.to_excel(f'Calls DataFrame from {ticker}.xlsx')
            puts_df.to_excel(f'Puts DataFrame from {ticker}.xlsx')
            print(f'{ticker} Options DataFrames Saved')
            user = False
        elif choice == 'e':
            print('You will exit')
            user = False
        else:
            print("Please just print 'a', 'b', 'c', 'd' or 'e'. You will exit")

user_experience()
