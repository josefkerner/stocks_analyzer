from typing import List
from fundamental.data_model.signal import Signal
class SignalEvaluator:
    def __init__(self):
        pass

    def evaluate(self, signals: List[Signal]):
        pass

    def get_stock_returns(self, signals: List[Signal]):
        balance = 1000
        portfolio = {

        }

        unique_tickers = list(set([signal.ticker for signal in signals]))
        for ticker in unique_tickers:
            signals = [signal for signal in signals if signal.ticker == ticker]
            #order signals by date
            signals = sorted(signals, key=lambda x: x.date)
            for signal in signals:
                if 'BUY' in signal.signal_type:
                    if balance > signal.price:
                        if signal.ticker not in portfolio:
                            portfolio[signal.ticker] = signal.price
                            balance -= signal.price
                            print(f'Buying {signal.ticker} for {signal.price}')
                elif 'SELL' in signal.signal_type:
                    if signal.ticker in portfolio:
                        del portfolio[signal.ticker]
                        balance += signal.price
                        print(f'Selling {signal.ticker} for {signal.price}')
        print(f'Balance: {balance}', portfolio)