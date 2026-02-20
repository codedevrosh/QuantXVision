def sma(series, window=20):
    return series.rolling(window).mean()