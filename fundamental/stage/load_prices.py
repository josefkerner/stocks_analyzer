from fundamental.lib.price_loader import PriceLoader
from datetime import datetime

def load_prices():
    price_loader = PriceLoader()
    today = datetime.now().today().strftime('%Y-%m-%d')


    tickers = ['AAPL','MSFT','GOOG','AMZN','NVDA','META','PWR','ACGL','CRWD','SPGI']
    for ticker in tickers:
        price_loader.load_price(
            ticker=ticker,
            start_dt='2022-01-01',
            end_dt=today
        )

load_prices()