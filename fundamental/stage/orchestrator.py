from fundamental.stage.load_prices import load_prices
from fundamental.stage.signaler import Signaler
from fundamental.stage.tech_ews import TechnicalEWS

class Orchestrator:
    def __init__(self):

        self.signaler = Signaler()
        self.tech_ews = TechnicalEWS()

    def run(self):
        load_prices()
        self.signaler.create_tech_signals()
        self.tech_ews.get_signals()
