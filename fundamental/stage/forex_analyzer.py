from pydantic import BaseModel

class InvestmentVariant(BaseModel):
    name: str
    duration: int
    income: float

    def to_dict(self):
        return {
            'name': self.name,
            'duration': f"{self.duration} year",
            'income': self.income
        }

class ForexAnalyzer:
    EUR_CZK = 24.68
    USD_CZK = 23.44
    EUR_USD = 1.05
    CZK_DEPOSITS = 3000000
    CZK_RATES = [0.06,0.04,0.02]
    EUR_CZK_ONE_YEAR_VARIANTS = [25.2, 25.0, 25.4]
    USD_CZK_ONE_YEAR_VARIANTS = [24.44, 24.1, 23.94]

    EUR_RATE = 0.0365
    USD_RATE = 0.0477
    def __init__(self):
        self.investment_variants = []

    def calc_investment_variants(self):
        self.get_czk_interest_income()
        self.calculate_eur_czk_swing_trade()
        self.calculate_usd_czk_swing_trade()

        for variant in self.investment_variants:
            print(variant.to_dict())



    def get_czk_interest_income(self):
        for rate in self.CZK_RATES:
            interest_income = self.CZK_DEPOSITS*(1+rate)

            name = f"CZK interest with rate of {rate}"
            self.investment_variants.append(InvestmentVariant(name=name,
                                                              duration=1,
                                                              income=interest_income))


    def get_eur_interest_income(self):
        interest_income = (self.CZK_DEPOSITS/self.EUR_CZK)*(1+self.EUR_RATE)

        return interest_income

    def get_usd_interest_income(self):
        interest_income = (self.CZK_DEPOSITS/self.USD_CZK)* (1+self.USD_RATE)
        return interest_income

    def calculate_eur_czk_swing_trade(self):
        interest_income = self.get_eur_interest_income()
        for variant in self.EUR_CZK_ONE_YEAR_VARIANTS:
            name = f"EUR swing, sell CZK {self.EUR_CZK}, buy CZK {variant}, invest in EUR deposit with rate {self.EUR_RATE}"
            variant = InvestmentVariant(name=name,
                                duration=1,
                                income=interest_income*variant)
            self.investment_variants.append(variant)

    def calculate_usd_czk_swing_trade(self):
        interest_income = self.get_usd_interest_income()
        for variant in self.USD_CZK_ONE_YEAR_VARIANTS:
            name = f"USD swing, sell CZK {self.USD_CZK}, inv, buy CZK {variant}, invest in USD deposit with rate {self.USD_RATE}"
            variant = InvestmentVariant(
                name=name,
                duration=1,
                income=interest_income*variant
            )
            self.investment_variants.append(variant)

if __name__ == '__main__':
    analyzer = ForexAnalyzer()
    analyzer.calc_investment_variants()