from dataclasses import dataclass
@dataclass
class FinIndicators:
    ticker_name: str
    date: str
    pe_ratio: float
    peg_ratio: float

    def get_record(self):
        rec = {
            'ticker_name': self.ticker_name,
            'date': self.date,
            'pe_ratio': self.pe_ratio,
            'peg_ratio': self.peg_ratio
        }
        return rec
