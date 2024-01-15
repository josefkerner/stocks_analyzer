from fundamental.indicators.technical.tech_indicators import TechIndicators
from fundamental.lib.connectors.dbconnector import DBconnector
from fundamental.data_model.signal import Signal
from fundamental.lib.price_loader import PriceLoader
from typing import List
from datetime import datetime

class TechnicalEWS:
    def __init__(self):
        self.price_loader = PriceLoader()
        self.tech_indicators = TechIndicators()
        self.db_connector = DBconnector()

    def get_signals(self):
        min_price_date = '2020-01-01'
        query = f"""
            SELECT
            ticker,
            price,
            signal_type,
            strength,
            date,
            description 
             FROM signals
            --min show date for viz
            WHERE date >= '{min_price_date}'
        """
        min_ews_date = '2023-10-01'
        #string to datetime.date object
        min_ews_date = datetime.strptime(min_ews_date, '%Y-%m-%d').date()

        recs = self.db_connector.query(query=query)
        #get unique tickers
        tickers = list(set([rec['ticker'] for rec in recs]))
        for ticker in tickers:
            #get ticker signals
            ticker_signals : List[Signal] = [
                Signal(**rec)
                for rec in recs if rec['ticker'] == ticker
            ]

            #filter signals where date is newer than min_ews_date
            ticker_signals = [
                signal for signal in ticker_signals
                if signal.date >= min_ews_date
            ]
            if len(ticker_signals) == 0:
                continue
            #get ticker prices for tickers where signal date is newer than min_ews_date
            today = datetime.now().today().strftime('%Y-%m-%d')
            ticker_prices = self.price_loader.prices_from_db(
                ticker=ticker,
                start_dt=min_price_date,
                end_dt=today
            )
            filtered_signals = self.tech_indicators.filter_signals(
                signals=ticker_signals,
                df=ticker_prices
            )
            if len(filtered_signals) == 0:
                print(f'No filtered signals for {ticker}')
                continue
            self.tech_indicators.visualize_ticker_signals(
                ticker=ticker,
                df=ticker_prices,
                signals=filtered_signals
            )

if __name__ == '__main__':
    ews = TechnicalEWS()
    ews.get_signals()








