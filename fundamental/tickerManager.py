import requests
from bs4 import BeautifulSoup

class TickerManager:
    def __init__(self):
        self.url = 'https://finviz.com/screener.ashx?v=141&f=an_recom_buybetter,cap_smallover,fa_curratio_o1.5,fa_eps5years_pos,fa_estltgrowth_pos,fa_netmargin_pos,fa_pb_u4,fa_pe_u40,fa_pfcf_u20,fa_quickratio_o1,fa_roe_pos,fa_sales5years_pos,ta_beta_0.5to1.5,ta_rsi_nob60&ft=3&o=recom'


    def parse_url(self):
        res = requests.get(self.url)
        soup = BeautifulSoup(res.text, 'html.parser')
        table = soup.find_all('table', attrs={'class': 'screener-body-table'})
        tickers = []
        for t in table:
            ticker_name = 'APPL'
            tickers.append(ticker_name)

