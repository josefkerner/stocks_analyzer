import os
import pandas as pd
from fundamental.indicators.technical.tech_indicators import TechIndicators

indicators = TechIndicators()
def test_bollinger_bands():

    base_dir = os.getcwd()
    df = pd.read_excel(f"{base_dir}/tests/indicators/prices.xlsx")
    df,signals = indicators.get_bollinger_bands(df=df,ticker='MSFT')
    true_signals = 0
    for signal in signals:
        increased_volume = indicators.test_volume(
            df=df,
            signal_date=signal.date,
            days_offset=3
        )
        if increased_volume:
            print(signal)
            true_signals += 1

    print(f"True signals: {true_signals} out of {len(signals)}")
    assert len(signals) > 0
