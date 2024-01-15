import pandas as pd
from fundamental.lib.viz.plotter import Plotter
from fundamental.data_model.signal import Signal
from typing import List
import pandas_ta as ta
import numpy as np

class TechIndicators:

    def __init__(self):
        self.plotter = Plotter()
    def compute_signals(self, df:pd.DataFrame, ticker:str):
        '''
        Will compute signals
        :param df:
        :param ticker:
        :return:
        '''
        df, signals = self.get_mas(df=df)
        df,signals = self.stochastic_oscilator(df=df,ticker=ticker, k=14, d=3)
        df, signals = self.get_bollinger_bands(df=df, ticker=ticker)
        return df, signals

    def visualize_ticker_signals(self, ticker: str, df: pd.DataFrame, signals: List[Signal]):
        '''
        Will visualize Signals
        :param ticker:
        :param df:
        :return:
        '''
        ticker_signals = [signal for signal in signals if signal.ticker == ticker]
        if len(ticker_signals) == 0:
            print('no signals')
            return True

        fig = self.plotter.plot_lines(
            df=df,
            x='date',
            y_cols=['close'
                #, 'ma_7', 'ma_14', 'ma_30'
                    ],
            title=f'{ticker} Prices with signals'
        )
        ticker_signals_recs = [signal.to_rec() for signal in ticker_signals]
        signals_df = pd.DataFrame(ticker_signals_recs)
        #get unique signal types
        signal_types = list(set(signals_df['signal_type']))
        for signal_type in signal_types:
            signals_df_filtered = signals_df[signals_df['signal_type'] == signal_type]
            fig = self.plotter.add_markers(
                fig=fig,
                df=signals_df_filtered,
                x='date',
                y='price',
                title=signal_type
            )
        self.plotter.show(fig=fig)

    def filter_signals(self, signals: List[Signal], df: pd.DataFrame) -> List[Signal]:
        '''
        Will filter signals
        :param signals:
        :param df:
        :return:
        '''
        filtered_signals = []
        for signal in signals:
            #check if signal is valid
            if self.test_volume(df=df, signal_date=signal.date):
                #signal is valid
                filtered_signals.append(signal)
        return filtered_signals


    def test_volume(self, df: pd.DataFrame, signal_date: str,days_offset: int = 2):
        '''

        :param df: prices dataframe
        :param signal_date: date when signal occured
        :param days_offset: num of days before signal_date
        :return:
        '''
        avg_volume = df['volume'].mean()
        #datetime object from signal_date
        date = pd.to_datetime(signal_date)

        #filter only last days_offset days before the signal_date
        #date column to datetime format
        df['date'] = pd.to_datetime(df['date'])
        df = df[df['date'] >= date - pd.Timedelta(days=days_offset)]
        #up to signal_date
        df = df[df['date'] <= date]
        offset_volume = df['volume'].mean()
        if abs(offset_volume) > abs(avg_volume) * 1.10:
            #increased volume
            return True
        else:
            #not a significant signal
            return False

    def get_mas(self, df: pd.DataFrame):
        '''
        Will calculate moving averages
        :param df: prices dataframe

        :return:
        '''

        mas = [7,14,30]
        signals = []

        for ma in mas:
            df[f'ma_{ma}'] = df['close'].rolling(ma).mean()
        #show all columns
        pd.set_option('display.max_columns', None)

        #plot volume, date as index in seaborn library
        #sns.lineplot(data=df, x='date', y='volume')
        #plt.show()


        #show chart with close, ma_7 and ma_14


        #calc MA crossings buy as a absolute difference between ma_14 and ma_30 less than 0.1
        #df['ma_crossed']= (abs(df['ma_14'] - df['ma_30']) < 0.1)
        #df['ma_crossed_buy'] = (df['ma_7'] > df['ma_14']) & (df['ma_14'] > df['ma_30'])
        #df['ma_crossed_sell'] = (df['ma_7'] < df['ma_14']) & (df['ma_14'] < df['ma_30'])
        print(df)

        #filter where ma_crossed_buy is true on current day but false on previous day
        df['ma_crossed'] = False
        for i,row in enumerate(df.itertuples()):
            if i == 0:
                continue
            ma_14_avg = (row.ma_14 + df.iloc[i-1].ma_14)/2
            ma_30_avg = (row.ma_30 + df.iloc[i-1].ma_30)/2

            if abs(ma_14_avg - ma_30_avg) < 0.9:
                print(f'{row.date} {ma_14_avg} {ma_30_avg}')
                #df.at[i,'ma_crossed'] = True

                lag_one_ma_14 = df.iloc[i-1].ma_14
                lag_one_ma_30 = df.iloc[i-1].ma_30
                date_prev = df.iloc[i - 1].date
                signals_prev = [signal for signal in signals if
                                signal.date == date_prev and signal.ticker == row.ticker
                                ]
                if lag_one_ma_14 < lag_one_ma_30:
                    #faster EMA crossed slower EMA from below
                    #previous day

                    #check if previous day is already in signals

                    if len(signals_prev) == 0:

                        print(f'Buy signal for {row.ticker} on {row.date}')
                        signals.append(
                            Signal(
                                ticker=row.ticker,
                                price=row.close,
                                strength=5,
                                signal_type='MA_BUY',
                                date=row.date,
                                description=f'MA crossed for {row.ticker} on {row.date}'
                            )
                        )
                else:
                    if len(signals_prev) == 0:
                        signals.append(
                            Signal(
                                ticker=row.ticker,
                                price=row.close,
                                strength=5,
                                signal_type='MA_SELL',
                                date=row.date,
                                description=f'MA crossed for {row.ticker} on {row.date}'
                            )
                        )
        return df, signals




    def stochastic_oscilator(self, df: pd.DataFrame, ticker: str, k:int, d: int):
        '''
        Will calculate stochastic oscilator
        :param df:
        :param ticker:
        :return:
        '''
        k_period = k
        d_period = d
        signals = []
        # Adds a "n_high" column with max value of previous 14 periods
        df['n_high'] = df['high'].rolling(k_period).max()
        # Adds an "n_low" column with min value of previous 14 periods
        df['n_low'] = df['low'].rolling(k_period).min()
        # Uses the min/max values to calculate the %k (as a percentage)
        df['%K'] = (df['close'] - df['n_low']) * 100 / (df['n_high'] - df['n_low'])
        # Uses the %k to calculates a SMA over the past 3 values of %k
        df['%D'] = df['%K'].rolling(d_period).mean()

        df.ta.stoch(high='high', low='low', k=14, d=3, append=True)
        k= df['%K'].iloc[-1]
        d = df['%D'].iloc[-1]

        def get_stochastic_levels(row):

            k = row['STOCHk_14_3_3']
            d = row['STOCHd_14_3_3']
            if k > 80 and d > 80 and k < d:
                return 'OVERBOUGHT'
            elif k < 20 and d < 20 and k > d:
                return 'OVERSOLD'
            else:
                return 'NORMAL'

        df['stoch_levels'] = df.apply(get_stochastic_levels, axis=1)
        #where stoch_levels != 'NORMAL'
        signals_df = df[df['stoch_levels'] != 'NORMAL']
        #sort by date
        signals_df = signals_df.sort_values(by='date', ascending=True)
        #get dates
        dates = signals_df['date'].unique()
        for i,row in enumerate(signals_df.to_dict(orient='records')):
            #get level for previous day
            if i == 0:
                continue

            if row['stoch_levels'] == 'OVERBOUGHT':
                #get previous row stoch_levels
                prev_row_levels = signals_df.iloc[i-1]['stoch_levels']
                if row['stoch_levels'] == prev_row_levels:
                    continue


                signal = Signal(
                    signal_type='Stochastic SELL',
                    date=row['date'],
                    ticker=ticker,
                    description=f'STOCHASTIC sell, k: {k}, d: {d}',
                    price=row['close'],
                    strength=5
                )
                signals.append(signal)
            elif row['stoch_levels'] == 'OVERSOLD':
                prev_row_levels = signals_df.iloc[i - 1]['stoch_levels']
                if row['stoch_levels'] == prev_row_levels:
                    continue
                signal = Signal(
                    signal_type='Stochastic BUY',
                    date=row['date'],
                    ticker=ticker,
                    description=f'STOCHASTIC buy, k: {k}, d: {d}',
                    price=row['close'],
                    strength=5
                )
                signals.append(signal)

        return df, signals

    def get_bollinger_bands(self, df: pd.DataFrame, ticker: str):
        '''
        Will calculate bollinger bands
        :param df:
        :param ticker:
        :return:
        '''

        def get_bollinger_signals(row)->str:
            '''
            Determines if price is above or below bollinger bands
            when the price is above upper bollinger band, return SELL
            when the price is below lower bollinger band, return BUY
            :param row:
            :return:
            '''
            print(row['close'], row['BBU_5_2.0'], row['BBL_5_2.0'])
            if abs(row['close'] - row['BBU_5_2.0']) < 1:
                return 'SELL'
            elif abs(row['close'] - row['BBL_5_2.0']) < 1:
                return 'BUY'
            else:
                return 'NORMAL'
        signals = []
        df.ta.bbands(append=True)
        df['bollinger_signal'] = df.apply(get_bollinger_signals, axis=1)
        #where bollinger_signal != 'NORMAL'
        signals_df = df
        #sort by date
        signals_df = signals_df.sort_values(by='date', ascending=True)
        for i,row in enumerate(signals_df.to_dict(orient='records')):
            #get signal of previous row
            if i == 0:
                continue
            prev_row_signal = signals_df.iloc[i-1]['bollinger_signal']
            if row['bollinger_signal'] == "NORMAL":
                continue
            if prev_row_signal != "NORMAL":
                continue
            signal = Signal(
                signal_type='Bollinger Bands',
                date=row['date'],
                ticker=ticker,
                description=f'Bollinger Bands {row["bollinger_signal"]}',
                price=row['close'],
                strength=5
            )
            signals.append(signal)
        return df, signals

    def get_lines_approach(self, df: pd.DataFrame):
        '''
        Indicate if price approaches support or resistence lines
        :return:
        '''
        prices = np.array(df["close"])
        K = 6
        kmeans = KMeans(n_clusters=6).fit(prices.reshape(-1, 1))
        # predict which cluster each price is in
        clusters = kmeans.predict(prices.reshape(-1, 1))

        # Create list to hold values, initialized with infinite values
        min_max_values = []
        # init for each cluster group
        for i in range(6):
            # Add values for which no price could be greater or less
            min_max_values.append([np.inf, -np.inf])

        # Get min/max for each cluster
        for i in range(len(prices)):
            # Get cluster assigned to price
            cluster = clusters[i]
            # Compare for min value
            if prices[i] < min_max_values[cluster][0]:
                min_max_values[cluster][0] = prices[i]
            # Compare for max value
            if prices[i] > min_max_values[cluster][1]:
                min_max_values[cluster][1] = prices[i]

        # Create container for combined values
        output = []
        # Sort based on cluster minimum
        s = sorted(min_max_values, key=lambda x: x[0])
        # For each cluster get average of
        for i, (_min, _max) in enumerate(s):
            # Append min from first cluster
            if i == 0:
                output.append(_min)
            # Append max from last cluster
            if i == len(min_max_values) - 1:
                output.append(_max)
            # Append average from cluster and adjacent for all others
            else:
                output.append(sum([_max, s[i + 1][0]]) / 2)