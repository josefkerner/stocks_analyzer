from bs4 import BeautifulSoup
import requests
class FundamentScreener:
    def __init__(self):

        page = requests.get('https://bulios.com/fair-prices')

        self.soup = BeautifulSoup(page.content, 'html.parser')

        #get rows from table that has class "simple-table"
        self.table = self.soup.find('table', {'class': 'simple-table'})
        self.rows = self.table.find_all('tr')
        for row in self.rows:
            #parse text of all cells in row
            cells = row.find_all('td')
            for cell in cells:
                text = str(cell.text).strip()
                #remove spaces and newlines
                text = text.replace('\n', '')
                print(text)

FundamentScreener()