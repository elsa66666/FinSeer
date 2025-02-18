import pandas as pd
import numpy as np


def calculate_ema(prices, span):
    return prices.ewm(span=span, adjust=False).mean()


def calculate_macd(df, slow=26, fast=12, signal=9):
    df1 = pd.DataFrame()
    df1['EMA_Fast'] = calculate_ema(df['adj_close'], fast)
    df1['EMA_Slow'] = calculate_ema(df['adj_close'], slow)
    df1['MACD'] = df1['EMA_Fast'] - df1['EMA_Slow']
    df1['Signal_Line'] = calculate_ema(df1['MACD'], signal)
    df['MACD_Histogram'] = df1['MACD'] - df1['Signal_Line']  # 1

    df1['EMA_Signal'] = np.where(df1['EMA_Fast'] > df1['EMA_Slow'], 1, -1)
    df1['EMA_Signal_Change'] = df1['EMA_Signal'].diff()
    conditions = [
        df1['EMA_Signal_Change'] == 2,
        df1['EMA_Signal_Change'] == -2
    ]
    choices = [
        'bullish_cross',
        'bearish_cross'
    ]
    df['macd_crossover'] = np.select(conditions, choices, default=None)  # 1
    return df


def calculate_bollinger_bands(df, window=20):
    df1 = pd.DataFrame()
    df1['Middle_Band'] = df['adj_close'].rolling(window=window).mean()
    df1['STD'] = df['adj_close'].rolling(window=window).std()
    df1['Upper_Band'] = df1['Middle_Band'] + (df1['STD'] * 2)
    df1['Lower_Band'] = df1['Middle_Band'] - (df1['STD'] * 2)
    conditions = [
        df['adj_close'] > df1['Upper_Band'],  # 1.差值
        df['adj_close'] < df1['Lower_Band']
    ]
    choices = [
        'exceeding_upper',
        'exceeding_lower'
    ]
    df['bollinger_bands'] = np.select(conditions, choices, default=None)
    df['exceeding_upper'] = np.select([df['adj_close'] > df1['Upper_Band']], [df['adj_close'] - df1['Upper_Band']], default=None)
    df['exceeding_lower'] = np.select([df['adj_close'] < df1['Lower_Band']], [df['adj_close'] - df1['Lower_Band']], default=None)
    return df


def calculate_kdj(df, n=9):
    df1 = pd.DataFrame()
    df1['Low_n'] = df['low'].rolling(window=n).min()
    df1['High_n'] = df['high'].rolling(window=n).max()
    df1['RSV'] = (df['close'] - df1['Low_n']) / (df1['High_n'] - df1['Low_n']) * 100

    df1['K'] = df1['RSV'].ewm(alpha=1 / 3, adjust=False).mean()
    df1['D'] = df1['K'].ewm(alpha=1 / 3, adjust=False).mean()
    df1['J'] = 3 * df1['K'] - 2 * df1['D']

    conditions1 = [
        (df1['K'] > 80) & (df1['D'] > 70) & (df1['J'] > 90),
        (df1['K'] < 20) & (df1['D'] < 30)
    ]
    choices1 = [
        'overbought_area',
        'oversold_area'
    ]
    df['overbought_and_oversold_conditions'] = np.select(conditions1, choices1, default=None)

    conditions2 = [
        df1['K'] > df1['D'],
        df1['K'] < df1['D']
    ]
    choices2 = [
        'bullish_signal',
        'bearish_signal'
    ]
    df['kdj_crossover'] = np.select(conditions2, choices2, default=None)
    return df


# czy
# From Selected_Alpha.ipynb
def calculate_returns(df):
    df['Returns'] = df['adj_close'].pct_change()
    return df


def calculate_vwap(df):
    df['VWAP'] = ((df['high'] + df['low'] + df['close']) / 3 * df['volume']) / df['volume']
    return df


def rank_series(s):
    """Return the rank of a pandas Series as percentile rank."""
    return s.rank(pct=True)


def ts_lag(df: pd.DataFrame, t: int = 1) -> pd.DataFrame:
    """Return the lagged values t periods ago.

    Args:
        :param df: tickers in columns, sorted dates in rows.
        :param t: lag

    Returns:
        pd.DataFrame: the lagged values
    """
    return df.shift(t)


def ts_delta(df, period=1):
    """
    Wrapper function to estimate difference.
    :param df: a pandas DataFrame.
    :param period: the difference grade.
    :return: a pandas DataFrame with today’s value minus the value 'period' days ago.
    """
    return df.diff(period)


def ts_sum(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Computes the rolling ts_sum for the given window size.

    Args:
        df (pd.DataFrame): tickers in columns, dates in rows.
        window      (int): size of rolling window.

    Returns:
        pd.DataFrame: the ts_sum over the last 'window' days.
    """
    return df.rolling(window).sum()


def ts_mean(df, window=10):
    """Computes the rolling mean for the given window size.

    Args:
        df (pd.DataFrame): tickers in columns, dates in rows.
        window      (int): size of rolling window.

    Returns:
        pd.DataFrame: the mean over the last 'window' days.
    """
    return df.rolling(window).mean()


# def ts_weighted_mean(df, period=10):
    """
    Linear weighted moving average implementation.
    :param df: a pandas DataFrame.
    :param period: the LWMA period
    :return: a pandas DataFrame with the LWMA.
    """
    # return (df.apply(lambda x: WMA(x, timeperiod=period)))


def ts_std(df, window=10):
    """
    Wrapper function to estimate rolling standard deviation.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    return (df
            .rolling(window)
            .std())


def ts_rank(df, window=10):
    """
    Wrapper function to estimate rolling rank.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series rank over the past window days.
    """
    return (df
            .rolling(window)
            .apply(lambda x: x.rank().iloc[-1]))


def ts_product(df, window=10):
    """
    Wrapper function to estimate rolling ts_product.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series ts_product over the past 'window' days.
    """
    return (df
            .rolling(window)
            .apply(np.prod))


def ts_min(df, window=10):
    """
    Wrapper function to estimate rolling min.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    return df.rolling(window).min()


def ts_max(df, window=10):
    """
    Wrapper function to estimate rolling min.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series max over the past 'window' days.
    """
    return df.rolling(window).max()


def ts_argmax(df, window=10):
    """
    Wrapper function to estimate which day ts_max(df, window) occurred on
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: well.. that :)
    """
    return df.rolling(window).apply(np.argmax).add(1)


def ts_argmin(df, window=10):
    """
    Wrapper function to estimate which day ts_min(df, window) occurred on
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: well.. that :)
    """
    return (df.rolling(window)
            .apply(np.argmin)
            .add(1))


def ts_corr(x, y, window=10):
    """
    Wrapper function to estimate rolling correlations.
    :param x, y: pandas DataFrames.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    return x.rolling(window).corr(y)


def ts_cov(x, y, window=10):
    """
    Wrapper function to estimate rolling covariance.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    return x.rolling(window).cov(y)


def add_mean_reversion_alpha(df):
    """
    Calculate and add a simple mean-reversion alpha to the DataFrame.
    The alpha is given by: -ln(today's open / yesterday's close)

    Parameters:
        df (pd.DataFrame): The DataFrame containing 'Open' and 'Close' columns.

    Returns:
        pd.DataFrame: The DataFrame with an additional 'alpha_smr' column.
    """
    # Calculate the mean-reversion alpha
    df['alpha_smr'] = -np.log(df['open'] / df['close'].shift(1))
    return df


def add_momentum_alpha(df):
    """
    Calculate and add a simple momentum alpha to the DataFrame.
    The alpha is given by: ln(yesterday's close / yesterday's open)

    Parameters:
        df (pd.DataFrame): The DataFrame containing 'Open' and 'Close' columns.

    Returns:
        pd.DataFrame: The DataFrame with an additional 'alpha_momentum' column.
    """
    # Calculate the momentum alpha based on the previous day's close and open
    df['alpha_mom'] = np.log(df['close'].shift(1) / df['open'].shift(1))
    return df


def alpha002(df):
    """Calculates alpha002 based on the formula:
       -1 * ts_corr(rank(ts_delta(log(volume), 2)), rank(((close - open) / open)), 6)"""

    # Necessary columns
    o = df['open']
    c = df['adj_close']
    v = df['volume']

    # Calculate components
    s1 = rank_series(ts_delta(np.log(v), 2))
    s2 = rank_series((c - o) / o)

    # Calculate the 6-day correlation and apply -1 multiplier
    alpha = -1 * ts_corr(s1, s2, 6)

    # Add alpha002 column to the DataFrame
    df['alpha002'] = alpha

    return df


def alpha006(df):
    # Ensure necessary columns are in the DataFrame
    if not {'volume', 'open'}.issubset(df.columns):
        raise ValueError("The input DataFrame must contain 'volume' and 'open' columns.")

    # Calculate correlation of Open and Volume over a 10-day window
    alpha_006_series = -1 * ts_corr(df['open'], df['volume'], window=10)

    # Add alpha006 column to the DataFrame
    df['alpha006'] = alpha_006_series

    return df


def alpha009(df):
    """(0 < ts_min(ts_delta(close, 1), 5)) ? ts_delta(close, 1)
    : ((ts_max(ts_delta(close, 1), 5) < 0)
    ? ts_delta(close, 1) : (-1 * ts_delta(close, 1)))
    """
    close_diff = ts_delta(df['adj_close'], 1)
    alpha = close_diff.where(ts_min(close_diff, 5) > 0,
                             close_diff.where(ts_max(close_diff, 5) < 0,
                                              -close_diff))

    df['alpha009'] = alpha
    return df


def alpha012(df):
    # Calculate delta(volume, 1) and delta(close, 1)
    v = df['volume']
    c = df['adj_close']

    # Calculate sign(delta(volume, 1)) * (-1 * delta(close, 1))
    alpha_012_series = np.sign(ts_delta(v, 1)).mul(-ts_delta(c, 1))

    # Add alpha012 column to the DataFrame
    df['alpha012'] = alpha_012_series

    return df


def alpha021(df):
    """ts_mean(close, 8) + ts_std(close, 8) < ts_mean(close, 2)
        ? -1
        : (ts_mean(close,2) < ts_mean(close, 8) - ts_std(close, 8)
            ? 1
            : (volume / adv20 < 1
                ? -1
                : 1))
    """
    c = df['adj_close']
    v = df['volume']
    sma2 = ts_mean(c, 2)
    sma8 = ts_mean(c, 8)
    std8 = ts_std(c, 8)

    cond_1 = sma8.add(std8) < sma2
    cond_2 = sma8.add(std8) > sma2
    cond_3 = v.div(ts_mean(v, 20)) < 1

    val = np.ones_like(c)
    alpha = np.where(cond_1, -1, np.where(cond_2, 1, np.where(cond_3, -1, 1)))
    df['alpha021'] = alpha

    return df


def alpha023(df):
    """Calculates alpha023 based on the specified condition logic:
       If the 20-day mean of 'High' is less than 'High', returns -1 * ts_delta(high, 2), else returns 0."""
    # Necessary columns
    h = df['high']

    # Calculate 20-day mean of High and 2-day delta of High
    H = ts_mean(h, 20)
    D = ts_delta(h, 2)

    # Apply condition to calculate alpha023
    alpha = np.where(H < h, -1 * D, 0)

    # Add alpha023 column to the DataFrame
    df['alpha023'] = alpha

    return df


def alpha024(df):
    """Calculates alpha024 based on the specified condition logic:
       If the 100-day delta of the 100-day mean of 'Close' divided by the lagged 'Close' is <= 0.05,
       returns -1 * (close - ts_min(close, 100)), else returns -1 * ts_delta(close, 3)."""

    # Necessary column
    c = df['adj_close']

    # Calculating condition components
    mean_100 = ts_mean(c, 100)
    delta_mean_100 = ts_delta(mean_100, 100)
    lag_close_100 = ts_lag(c, 100)

    # Condition
    cond = delta_mean_100 / lag_close_100 <= 0.05

    # Calculating values based on condition
    alpha = np.where(cond, -1 * (c - ts_min(c, 100)), -1 * ts_delta(c, 3))

    # Adding alpha024 column to the DataFrame
    df['alpha024'] = alpha

    return df


def alpha028(df):
    """scale(((ts_corr(adv20, low, 5) + (high + low) / 2) - close))"""
    h = df['high']
    l = df['low']
    c = df['adj_close']
    v = df['volume']
    adv20 = ts_mean(v, 20)

    # Calculate ts_corr(adv20, low, 5)
    corr_adv20_low = ts_corr(adv20, l, window=5)

    # Calculate the expression ((ts_corr(adv20, low, 5) + (high + low) / 2) - close)
    alpha_expression = (corr_adv20_low + (h + l) / 2) - c

    # Scale the result
    # Scaling function can be achieved by normalizing (mean=0, std=1) the expression
    alpha_scaled = (alpha_expression - alpha_expression.mean()) / alpha_expression.std()

    # Add alpha028 column to the DataFrame
    df['alpha028'] = alpha_scaled

    return df


def alpha032(df):
    """Calculates alpha032 based on the formula:
       scale(ts_mean(close, 7) - close) + (20 * scale(ts_corr(vwap, ts_lag(close, 5), 230)))"""

    # Necessary columns
    c = df['adj_close']
    vwap = ((df['high'] + df['low'] + df['close']) / 3 * df['volume']) / df['volume']

    # Calculate 7-day mean of close
    mean_close_7 = ts_mean(c, 7)

    # Calculate 5-day lag of close
    lag_close_5 = ts_lag(c, 5)

    # Calculate ts_corr(vwap, lagged close, 230)
    corr_vwap_lag_close = ts_corr(vwap, lag_close_5, 230)

    # Calculate the expression components
    first_term = mean_close_7 - c
    scaled_first_term = (first_term - first_term.mean()) / first_term.std()  # Scale first term

    scaled_corr = (
                              corr_vwap_lag_close - corr_vwap_lag_close.mean()) / corr_vwap_lag_close.std()  # Scale correlation term

    # Combine terms to calculate alpha032
    alpha = scaled_first_term + (20 * scaled_corr)

    # Add alpha032 column to the DataFrame
    df['alpha032'] = alpha

    return df


def alpha041(df):
    """(((high * low)^0.5) - vwap)"""

    # Necessary columns
    h = df['high']
    l = df['low']
    vwap = ((df['high'] + df['low'] + df['close']) / 3 * df['volume']) / df['volume']

    # Calculate the expression ((high * low)^0.5) - vwap
    alpha = (h * l) ** 0.5 - vwap

    # Add alpha041 column to the DataFrame
    df['alpha041'] = alpha

    return df


def alpha046(df):
    """0.25 < ts_lag(ts_delta(close, 10), 10) / 10 - ts_delta(close, 10) / 10
            ? -1
            : ((ts_lag(ts_delta(close, 10), 10) / 10 - ts_delta(close, 10) / 10 < 0)
                ? 1
                : -ts_delta(close, 1))
    """
    # Necessary column
    c = df['adj_close']

    # Calculate components
    delta_close_10 = ts_delta(c, 10)
    lagged_delta_close_10 = ts_lag(delta_close_10, 10)
    expression = lagged_delta_close_10.div(10) - delta_close_10.div(10)

    # Apply conditions
    alpha = np.where(expression > 0.25, -1, np.where(expression < 0, 1, -ts_delta(c, 1)))

    # Add alpha046 column to the DataFrame
    df['alpha046'] = alpha

    return df


def alpha049(df):
    """Calculates alpha049 based on the specified condition logic:
       If (lagged 10-day delta of close / 10 - 10-day delta of close / 10) < -0.1 * close, returns 1.
       Otherwise, returns -delta(close, 1)."""

    # Necessary column
    c = df['adj_close']

    # Calculate components
    delta_lagged_close_10 = ts_delta(ts_lag(c, 10), 10)
    delta_close_10 = ts_delta(c, 10)
    expression = delta_lagged_close_10.div(10) - delta_close_10.div(10)

    # Apply conditions
    alpha = np.where(expression < -0.1 * c, 1, -ts_delta(c, 1))

    # Add alpha049 column to the DataFrame
    df['alpha049'] = alpha

    return df


def alpha051(df):
    """Calculates alpha051 based on the specified condition logic:
       If (lagged 10-day delta of close / 10 - 10-day delta of close / 10) < -0.05 * close, returns 1.
       Otherwise, returns -delta(close, 1)."""

    # Necessary column
    c = df['adj_close']

    # Calculate components
    delta_lagged_close_10 = ts_delta(ts_lag(c, 10), 10)
    delta_close_10 = ts_delta(c, 10)
    expression = delta_lagged_close_10.div(10) - delta_close_10.div(10)

    # Apply conditions
    alpha = np.where(expression < -0.05 * c, 1, -ts_delta(c, 1))

    # Add alpha051 column to the DataFrame
    df['alpha051'] = alpha

    return df


def alpha053(df):
    """-1 * ts_delta(1 - (high - close) / (close - low), 9)"""
    # Necessary columns
    h = df['high']
    c = df["adj_close"]
    l = df['low']

    # Calculate the expression inside ts_delta
    expression = 1 - (h - c) / (c - l)

    # Calculate the 9-day delta of the expression and apply -1 multiplier
    alpha = -1 * ts_delta(expression, 9)

    # Add alpha053 column to the DataFrame
    df['alpha053'] = alpha

    return df


def alpha054(df):
    """((-1 * ((low - close) * (open^5))) / ((low - high) * (close^5)))"""

    o = df['open']
    h = df['high']
    l = df['low']
    c = df['adj_close']

    # Calculate the expression
    numerator = -1 * ((l - c) * (o ** 5))
    denominator = (l - h) * (c ** 5)
    alpha = numerator / denominator

    # Add alpha054 column to the DataFrame
    df['alpha054'] = alpha

    return df


def alpha101(df):
    """((close - open) / ((high - low) + 0.001))"""

    o = df['open']
    h = df['high']
    l = df['low']
    c = df['adj_close']

    # Calculate the expression
    alpha = (c - o) / ((h - l) + 0.001)

    # Add alpha101 column to the DataFrame
    df['alpha101'] = alpha

    return df
