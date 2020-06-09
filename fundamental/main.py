from fundamental import config
from fundamental import company_profiles as profiles
from fundamental import company_financials as financials
from fundamental import company_fundamentals as fundamentals

import pandas as pd


def prepare_company_profiles(price_filter, api_key):
    """
    Build company profiles for stock tickers that meet price and exchange requirements.
    Reference: https://financialmodelingprep.com/developer/docs/#Company-Profile

    :param price_filter: The minimum stock price a user is willing to consider
    :param api_key: FinancialModelingPrep API key
    :return: None
    """

    # Retrieve all available stock tickers from the FMP API
    data = profiles.get_company_data(api_key)
    # Drop rows with 1 or more null values (cols: "symbol", "name", "price", "exchange")
    data.dropna(axis=0, how='any', inplace=True)
    # Only retain stocks listed on major exchanges
    data = profiles.select_stock_exchanges(data)
    # Only retain stocks with a price greater than or equal to the provided price filter
    data = profiles.select_minimum_price(data, price_filter)
    # Retrieve company profile data for remaining stock tickers
    profiles.create_company_profile(data, api_key)

    return None


def get_company_financials(file_path, review_period, report_year, eval_period, dir_path, api_key):
    """
    Retrieve, clean, subset, and store company financial data in the specified directory.

    :param file_path: File path to company profile data generated by the function above
    :param review_period: Frequency of financial statements - 'annual' or 'quarter'
    :param report_year: Year of most recent financial report desired
    :param eval_period: Number of years prior to most recent report to be analyzed (max = 10)
    :param dir_path: Directory path where csv files should be stored
    :param api_key: FinancialModelingPrep API key
    :return: None
    """

    # Read in company profile data generated by the function above
    company_profiles = pd.read_csv(file_path)
    # Subset DataFrame to companies in the provided sectors (*args)
    sector_companies = financials.select_sector(company_profiles, 'Consumer Cyclical')
    # List financial data you wish to retrieve from the FMP API
    request_list = ['financials', 'financial-ratios', 'financial-statement-growth',
                    'company-key-metrics', 'enterprise-value']

    for request in request_list:
        # Retrieve financial data described in the list above on an annual or quarterly basis
        raw_data = financials.get_financial_data(sector_companies, request, review_period, api_key)
        # Remove rows with corrupted date values, create new year column
        clean_data = financials.clean_financial_data(raw_data)
        # Subset financial data to the date range provided
        subset_data = financials.select_analysis_years(clean_data, report_year, eval_period)
        # Save financial data to data directory for further analysis
        evaluation_period = subset_data.year.max() - subset_data.year.min()
        filename = dir_path + request + '-' + str(evaluation_period) + 'Y' + '.csv'
        subset_data.to_csv(filename, index=False, header=True)

        return None


if __name__ == '__main__':
    prepare_company_profiles(10.00, config.api_key)
    get_company_financials('data/company-profiles.csv', 'annual', 2019, 10, 'data/', config.api_key)