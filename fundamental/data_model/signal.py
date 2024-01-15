from dataclasses import dataclass
@dataclass
class Signal:
    ticker:str
    price: float
    signal_type: str
    strength: int
    date:str
    description: str

    def __init__(self,
                 ticker,
                    price,
                    signal_type,
                    strength,
                    date,
                    description
                 ):
        self.ticker = ticker
        self.price = price
        self.signal_type = signal_type
        self.strength = strength
        self.date = date
        self.description = description

    def to_rec(self):
        rec = {
            'ticker': self.ticker,
            'price': self.price,
            'signal_type': self.signal_type,
            'strength': self.strength,
            'date': self.date,
            'description': self.description
        }
        return rec