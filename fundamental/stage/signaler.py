import matplotlib.pyplot as plt
import pandas as pd
from typing import List
from fundamental.lib.price_loader import PriceLoader
from datetime import datetime
from dataclasses import dataclass
from fundamental.data_model.signal import Signal
from fundamental.indicators.technical.tech_indicators import TechIndicators

@dataclass


class Signaler:
    SIGNAL_TABLE = 'signals'
    def __init__(self):
        self.price_loader = PriceLoader()
        self.db_manager = self.price_loader.db_connector
        self.tech_indicators = TechIndicators()

        self.signals : List[Signal] = []

    def save_signals(self, signals: List[Signal]):
        '''
        Will save signals into DB
        :param signals:
        :return:
        '''
        signals_recs = [signal.to_rec() for signal in signals]
        self.db_manager.save_records(
            table_name=self.SIGNAL_TABLE,
            records=signals_recs
        )

    def create_tech_signals(self):
        query = f"""
        SELECT ticker FROM {self.price_loader.TBL_PRICES}
        GROUP BY ticker
        """
        recs = self.db_manager.query(query=query)
        #get unique tickers
        tickers = list(set([rec['ticker'] for rec in recs]))

        for ticker in tickers:
            print(f'Processing {ticker}')
            prices = self.price_loader.prices_from_db(
                ticker=ticker,
                start_dt='2020-01-01',
                end_dt=datetime.now().today().strftime('%Y-%m-%d')
            )
            if ticker == 'test':
                continue
            df, signals = self.tech_indicators.compute_signals(df=prices, ticker=ticker)
            self.save_signals(signals=signals)

if __name__ == '__main__':

    signaler = Signaler()
    signaler.create_tech_signals()


