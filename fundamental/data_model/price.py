from dataclasses import dataclass
@dataclass
class Price:
    ticker_name: str
    date: str
    day: int
    month: int
    year: int
    day_of_week: int
    high: float
    low: float
    close: float
    open: float
    volume: int

    def get_ma(self):
        pass

    def get_record(self):
        rec = {
            'ticker': self.ticker_name,
            'date': self.date,
            'day': self.day,
            'month': self.month,
            'year': self.year,
            'day_of_week': self.day_of_week,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'open': self.open,
            'volume': self.volume
        }
        return rec
