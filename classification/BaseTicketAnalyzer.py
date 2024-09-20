from abc import ABC, abstractmethod
import pandas as pd

class BaseTicketAnalyzer(ABC):
    def __init__(self, incident_number, description, db_connection):
        self.incident_number = incident_number
        self.description = description
        self.db_connection = db_connection
        self.order_number = None
        self.analysis = []

    @abstractmethod
    def extract_order_numbers(self):
        """Step 2: Extract order numbers from the description"""
        pass

    @abstractmethod
    def match_order_number(self):
        """Step 3: Match order numbers in t_raw_order"""
        pass

    @abstractmethod
    def fetch_master_order_details(self):
        """Step 4: Fetch master order details from t_master_order"""
        pass

    @abstractmethod
    def fetch_transaction_steps(self):
        """Step 5: Fetch work steps from t_transaction_manager"""
        pass

    @abstractmethod
    def analyze_work_steps(self):
        """Step 6: Analyze work steps for completion"""
        pass

    def run_analysis(self):
        """Main method to run the analysis across all steps"""
        self.extract_order_numbers()
        self.match_order_number()
        if self.order_number is not None:
            self.fetch_master_order_details()
            self.fetch_transaction_steps()
            self.analyze_work_steps()
        return self.analysis
