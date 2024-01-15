from lib.company_profiler import CompanyProfiler
from lib.ratioCalculator import RatioCalculator
import pandas as pd
class StockScreener:
    def __init__(self):
        self.API_KEY = '4e8ff111cf9ca9d128c5941a2bf0d671'
        self.company_profiler = CompanyProfiler(self.API_KEY)
        self.ratio_calculator = RatioCalculator()

    def get_profiles(self):
        import os

        tickers = ['MSFT']


        # pandas show all columns
        pd.set_option('display.max_columns', None)
        for ticker in tickers:
            if not os.path.exists(f'{ticker}_company_data.xlsx'):
                company_data = self.company_profiler.get_company_profile(ticker)
                #print(company_row)
                company_data['symbol'] = ticker
                company_data.to_excel(f'{ticker}_company_data.xlsx')

            else:
                company_data = pd.read_excel(f'{ticker}_company_data.xlsx')

            if not os.path.exists(f'{ticker}_financial_data.xlsx'):
                fin_data = self.company_profiler.get_company_data(ticker)
                fin_data['symbol'] = ticker
                fin_data.to_excel(f'{ticker}_financial_data.xlsx')
            else:
                fin_data = pd.read_excel(f'{ticker}_financial_data.xlsx')
            all_data = fin_data.merge(company_data, on='symbol',
                                         how='inner', suffixes=('', '_x'))

            all_data.to_excel(f'{ticker}_all_data.xlsx')
            self.get_calculated_metrics(all_data)



    def get_calculated_metrics(self, df: pd.DataFrame):


        report_year = 2022
        eval_period = 4
        stocks = ['MSFT']
        ttm_data = df[df.year == report_year]

        # Calculate the N-YEAR mean, median, or percent change of a given set of columns
        performance_stats = self.ratio_calculator.calculate_stats(df,
                                                                  stat='median',
                                                                  report_year=report_year,
                                                         eval_period=eval_period,
                                                      calculated_columns=['roe','currentRatio']
                                                                  )

        stock_growth_estimates = {'MSFT': 0.06}


        performance_stats.reset_index(drop=True,inplace=True)
        #ttm_data.reset_index(inplace=True)
        ttm_data['year'] = ttm_data['year'].astype(int)
        performance_stats['year'] = performance_stats['year'].astype(int)
        df = ttm_data.merge(performance_stats, on=['symbol', 'year'], how='inner', suffixes=('', '_x'))

        print('df combined-------------')
        print(df.head(2))


        if len(df) == 0:
            print('data combined is empty')

        df = self.ratio_calculator.prepare_valuation_inputs(df, report_year, eval_period, stocks=stocks)
        df = self.ratio_calculator.calculate_discount_rate(df=df)
        df = self.ratio_calculator.calculate_discounted_free_cash_flow(df=df,
                                                                       projection_window=5,
                                                                       **stock_growth_estimates
                                                                       )

        df = self.ratio_calculator.calculate_terminal_value(df=df)
        df = self.ratio_calculator.calculate_intrinsic_value(df=df)
        print(df.head(2))


if __name__ == '__main__':
    stock_screener = StockScreener()
    stock_screener.get_profiles()
