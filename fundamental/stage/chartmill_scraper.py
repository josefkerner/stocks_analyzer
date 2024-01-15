import pandas as pd
import requests
from selenium import webdriver
from bs4 import BeautifulSoup
import time
from selenium import webdriver
from selenium.webdriver import Chrome
from typing import Any, Dict, List, Optional, Tuple
from selenium.webdriver.common.by import By

import chromedriver_autoinstaller


chromedriver_autoinstaller.install()

# Define the Chrome webdriver options
options = webdriver.ChromeOptions()
options.add_argument("--headless") # Set the Chrome webdriver to run in headless mode for scalability

# By default, Selenium waits for all resources to download before taking actions.
# However, we don't need it as the page is populated with dynamically generated JavaScript code.
options.page_load_strategy = "none"

# Pass the defined options objects to initialize the web driver
driver = Chrome(options=options)
# Set an implicit wait of 5 seconds to allow time for elements to appear before throwing an exception
driver.implicitly_wait(5)



class ChartMillScraper:
    def __init__(self):
        df = self.get_data()
        print(df)


    def get_data(self):
        url = "https://www.chartmill.com/stock/stock-screener?ia4=1&s=t&nw=1&o1=3&op1=200,16711680&o2=3&op2=50,255&o3=1&v=19&f=sl_pio_8_X,sl_val_6_X,v1_50b500,p_pg5,mc_smp,exch_us&sd=ASC"
        #get html content with enabled javascript via a selenium driver
        html_content= driver.get(url)
        time.sleep(5)
        #print(driver.page_source)
        #parse html content using beautiful soup
        soup = BeautifulSoup(driver.page_source, "html.parser")
        #get table with cdk table
        #print(soup)
        table = soup.find("table", attrs={"class": "cdk-table"})
        #print(table)
        #get table header columns
        table_header = table.find("thead")
        #print(table_header)
        table_header_columns = table_header.find_all("th")
        cols_text = [col.text for col in table_header_columns]
        #strip all whitespaces from the text
        cols_text = [col.strip() for col in cols_text]
        #replace whitespaces with underscores
        cols_text = [col.replace(" ", "_") for col in cols_text]
        #make all text lowercase
        cols_text = [col.lower() for col in cols_text]

        print(cols_text)
        rows : List[Dict[str, Any]] = []
        for row in table.find("tbody").find_all("tr"):
            row_dict = {}
            for i, cell_name in enumerate(cols_text):
                #get text of all elements in the cell
                cell = row.find_all("td")[i]
                text = [text for text in cell.stripped_strings]
                text = " ".join(text)
                text = text.strip()
                if cell_name == "dividend_yield":
                    if text == "N/A":
                        text = 0.0
                #try to convert text to float
                try:
                    text = float(text)
                except:
                    pass

                row_dict[table_header_columns[i].text] = text
            rows.append(row_dict)
        #convert rows to dataframe
        df = pd.DataFrame(rows)
        #add a column with the current date
        df['date'] = pd.to_datetime('today').date()
        print(rows)
        return df

ChartMillScraper().get_data()