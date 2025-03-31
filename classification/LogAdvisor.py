import streamlit as st
import requests
import json
import pandas as pd
import logging
from datetime import datetime
import os
from dotenv import load_dotenv
import sqlite3
from elasticsearch import Elasticsearch
import google.generativeai as genai

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("log_advisor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("LogAdvisor")

# Load environment variables
load_dotenv()

# Initialize database connection
def get_db_connection():
    """Create and return a database connection"""
    conn = sqlite3.connect('log_advisor.db')
    conn.row_factory = sqlite3.Row
    return conn

# Initialize Elasticsearch client
def get_es_client():
    """Create and return Elasticsearch client"""
    es_url = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
    es_user = os.getenv("ELASTICSEARCH_USER")
    es_pass = os.getenv("ELASTICSEARCH_PASSWORD")
    
    if es_user and es_pass:
        return Elasticsearch(es_url, basic_auth=(es_user, es_pass))
    else:
        return Elasticsearch(es_url)

# Initialize LLM client
def get_llm_client():
    """Create and return LLM client (Google Gemini in this example)"""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("No Gemini API key found. Set GEMINI_API_KEY in .env file")
        return None
    
    genai.configure(api_key=api_key)
    return genai

class EnrichmentService:
    """Handle transaction enrichment via external APIs"""
    
    def __init__(self, db_conn):
        self.conn = db_conn
        self.logger = logging.getLogger("LogAdvisor.EnrichmentService")
    
    def get_enrichment_api_details(self, app_id):
        """Get API details for the given app_id"""
        cursor = self.conn.cursor()
        query = """
            SELECT * FROM AppEnrichmentAPI 
            WHERE logging_app_id = (
                SELECT app_id FROM LoggingApps WHERE app_id = ?
            )
        """
        cursor.execute(query, (app_id,))
        return cursor.fetchone()
    
    def enrich_transaction(self, app_id, transaction_id):
        """Enrich transaction by calling the appropriate API"""
        self.logger.info(f"Enriching transaction {transaction_id} for app {app_id}")
        
        api_details = self.get_enrichment_api_details(app_id)
        if not api_details:
            self.logger.warning(f"No enrichment API found for app_id {app_id}")
            return None
        
        # In a real implementation, you would use the API details to make the actual API call
        # For this PoC, we'll simulate an API response
        
        # Mock API call
        if api_details['api_type'] == 'rest':
            try:
                # In production, use actual endpoint URL and request schema
                # For demo, just simulating a response
                simulated_response = {
                    "transaction_id": transaction_id,
                    "timestamp": datetime.now().isoformat(),
                    "customer_id": "XXXX1234",  # Masked sensitive data
                    "transaction_type": "Payment",
                    "status": "Completed",
                    "amount": "XXX.XX",  # Masked sensitive data
                    "region": "North America",
                    "channel": "Mobile"
                }
                self.logger.info(f"Successfully enriched transaction {transaction_id}")
                return simulated_response
            except Exception as e:
                self.logger.error(f"Error enriching transaction: {str(e)}")
                return None
        else:
            self.logger.warning(f"API type {api_details['api_type']} not implemented")
            return None

class LogSearchService:
    """Handle searching logs in Elasticsearch"""
    
    def __init__(self, es_client, db_conn):
        self.es = es_client
        self.conn = db_conn
        self.logger = logging.getLogger("LogAdvisor.LogSearchService")
    
    def get_log_source(self, app_id):
        """Get log source for the given app_id"""
        cursor = self.conn.cursor()
        query = "SELECT log_source FROM LoggingApps WHERE app_id = ?"
        cursor.execute(query, (app_id,))
        result = cursor.fetchone()
        return result['log_source'] if result else None
    
    def search_logs(self, app_id, transaction_id, start_time, end_time):
        """Search logs in Elasticsearch"""
        self.logger.info(f"Searching logs for transaction {transaction_id} in app {app_id}")
        
        log_source = self.get_log_source(app_id)
        if not log_source:
            self.logger.warning(f"No log source found for app_id {app_id}")
            return []
        
        try:
            # Build Elasticsearch query
            query = {
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"transaction_id": transaction_id}},
                            {
                                "range": {
                                    "@timestamp": {
                                        "gte": start_time,
                                        "lte": end_time
                                    }
                                }
                            }
                        ]
                    }
                },
                "sort": [{"@timestamp": {"order": "asc"}}]
            }
            
            # In a real implementation, you would use the actual index name
            # For this PoC, we'll simulate Elasticsearch response
            
            # Mock Elasticsearch search
            # In production this would be: response = self.es.search(index=log_source, body=query)
            
            # Simulated logs
            simulated_logs = [
                {
                    "@timestamp": "2023-03-15T10:15:32.345Z",
                    "level": "INFO",
                    "message": f"Transaction {transaction_id} started processing",
                    "transaction_id": transaction_id,
                    "service": "payment-processor"
                },
                {
                    "@timestamp": "2023-03-15T10:15:33.123Z",
                    "level": "INFO",
                    "message": f"Customer data retrieved for transaction {transaction_id}",
                    "transaction_id": transaction_id,
                    "service": "customer-service"
                },
                {
                    "@timestamp": "2023-03-15T10:15:34.987Z",
                    "level": "WARN",
                    "message": f"Slow database response for transaction {transaction_id}",
                    "transaction_id": transaction_id,
                    "service": "database-service",
                    "response_time_ms": 1500
                },
                {
                    "@timestamp": "2023-03-15T10:15:36.234Z",
                    "level": "ERROR",
                    "message": f"Timeout occurred for transaction {transaction_id} when calling payment gateway",
                    "transaction_id": transaction_id,
                    "service": "payment-gateway",
                    "error_code": "TIMEOUT_ERROR"
                },
                {
                    "@timestamp": "2023-03-15T10:15:40.123Z",
                    "level": "INFO",
                    "message": f"Retry successful for transaction {transaction_id}",
                    "transaction_id": transaction_id,
                    "service": "payment-gateway"
                },
                {
                    "@timestamp": "2023-03-15T10:15:42.456Z",
                    "level": "INFO",
                    "message": f"Transaction {transaction_id} completed successfully",
                    "transaction_id": transaction_id,
                    "service": "payment-processor"
                }
            ]
            
            self.logger.info(f"Found {len(simulated_logs)} logs for transaction {transaction_id}")
            return simulated_logs
            
        except Exception as e:
            self.logger.error(f"Error searching logs: {str(e)}")
            return []

class LogAnalysisService:
    """Analyze logs using LLM"""
    
    def __init__(self, llm_client):
        self.llm = llm_client
        self.logger = logging.getLogger("LogAdvisor.LogAnalysisService")
    
    def analyze_logs(self, transaction_data, logs):
        """Analyze logs using LLM"""
        if not self.llm:
            self.logger.error("LLM client not initialized")
            return "Error: LLM service unavailable. Please check API key configuration."
        
        self.logger.info(f"Analyzing logs for transaction {transaction_data.get('transaction_id', 'unknown')}")
        
        # Prepare context for LLM
        transaction_context = f"""
        Transaction ID: {transaction_data.get('transaction_id', 'N/A')}
        Transaction Type: {transaction_data.get('transaction_type', 'N/A')}
        Status: {transaction_data.get('status', 'N/A')}
        Region: {transaction_data.get('region', 'N/A')}
        Channel: {transaction_data.get('channel', 'N/A')}
        """
        
        log_messages = "\n".join([
            f"[{log.get('@timestamp', 'N/A')}] [{log.get('level', 'N/A')}] {log.get('message', 'N/A')}"
            for log in logs
        ])
        
        prompt = f"""
        As a log analysis assistant, analyze the following transaction logs and provide a human-readable summary.
        Focus on explaining the underlying conditions, any issues observed, and the final outcome.
        DO NOT include any sensitive data in your analysis.
        
        Transaction Information:
        {transaction_context}
        
        Log Messages:
        {log_messages}
        
        Provide a concise analysis that explains:
        1. What happened with this transaction
        2. Any issues or errors encountered
        3. How the issues were resolved (if they were)
        4. The final outcome of the transaction
        """
        
        try:
            # In a real implementation, you would use the actual LLM API
            # For this PoC, we'll simulate LLM response
            
            # For Gemini, this would be:
            # model = self.llm.GenerativeModel('gemini-pro')
            # response = model.generate_content(prompt)
            # analysis = response.text
            
            # Simulated LLM analysis
            analysis = """
            Transaction Analysis:
            
            This payment transaction was processed through the mobile channel in the North America region. The transaction went through the following stages:
            
            1. The transaction was initiated and started processing in the payment-processor service.
            2. Customer data was successfully retrieved from the customer-service.
            3. A warning was recorded due to slow database response (1500ms), which is above the normal threshold.
            4. An error occurred when calling the payment gateway, specifically a timeout error.
            5. The system automatically retried the payment gateway call, which was successful.
            6. The transaction was eventually completed successfully.
            
            The main issue encountered was a timeout when communicating with the payment gateway, likely due to network latency or high load on the gateway. The system's retry mechanism worked as designed, allowing the transaction to complete successfully despite the initial failure.
            
            Recommendation: Monitor the database response times and payment gateway timeouts. If these issues persist, consider investigating potential performance bottlenecks in these services.
            """
            
            self.logger.info("Successfully analyzed logs with LLM")
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing logs with LLM: {str(e)}")
            return f"Error analyzing logs: {str(e)}"

class ApplicationService:
    """Main application service that coordinates the workflow"""
    
    def __init__(self):
        self.db_conn = get_db_connection()
        self.es_client = get_es_client()
        self.llm_client = get_llm_client()
        
        self.enrichment_service = EnrichmentService(self.db_conn)
        self.log_search_service = LogSearchService(self.es_client, self.db_conn)
        self.log_analysis_service = LogAnalysisService(self.llm_client)
        
        self.logger = logging.getLogger("LogAdvisor.ApplicationService")
    
    def get_applications(self):
        """Get list of applications from database"""
        cursor = self.db_conn.cursor()
        cursor.execute("SELECT app_id, app_name FROM LoggingApps")
        return [dict(row) for row in cursor.fetchall()]
    
    def process_transaction(self, app_id, transaction_id, start_time, end_time):
        """Process transaction workflow"""
        self.logger.info(f"Processing transaction {transaction_id} for app {app_id}")
        
        # Step 1: Enrich transaction
        transaction_data = self.enrichment_service.enrich_transaction(app_id, transaction_id)
        if not transaction_data:
            return {"error": "Failed to enrich transaction"}
        
        # Step 2: Search logs
        logs = self.log_search_service.search_logs(app_id, transaction_id, start_time, end_time)
        if not logs:
            return {"error": "No logs found for the transaction"}
        
        # Step 3: Analyze logs
        analysis = self.log_analysis_service.analyze_logs(transaction_data, logs)
        
        return {
            "transaction_data": transaction_data,
            "logs": logs,
            "analysis": analysis
        }
    
    def close(self):
        """Close database connection"""
        if self.db_conn:
            self.db_conn.close()

# Initialize database schema (for demo purposes)
def init_database():
    """Initialize database schema and sample data"""
    conn = sqlite3.connect('log_advisor.db')
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS LoggingApps (
        app_id INTEGER PRIMARY KEY,
        log_source TEXT NOT NULL,
        app_name TEXT NOT NULL,
        description TEXT
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS LoggingAppFields (
        field_id INTEGER PRIMARY KEY,
        logging_app_id INTEGER NOT NULL,
        app_transaction_id TEXT NOT NULL,
        app_transaction_id_description TEXT,
        FOREIGN KEY (logging_app_id) REFERENCES LoggingApps (app_id)
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS AppEnrichmentAPI (
        app_api_id INTEGER PRIMARY KEY,
        logging_app_id INTEGER NOT NULL,
        endpoint_name TEXT NOT NULL,
        endpoint_url TEXT NOT NULL,
        api_type TEXT NOT NULL,
        request_schema TEXT,
        response_schema TEXT,
        FOREIGN KEY (logging_app_id) REFERENCES LoggingApps (app_id)
    )
    ''')
    
    # Insert sample data
    cursor.execute('''
    INSERT OR IGNORE INTO LoggingApps (app_id, log_source, app_name, description)
    VALUES 
        (1, 'payment-logs', 'Payment Processing System', 'Handles all payment transactions'),
        (2, 'customer-logs', 'Customer Management System', 'Manages customer data and interactions'),
        (3, 'inventory-logs', 'Inventory Management System', 'Tracks product inventory and movement')
    ''')
    
    cursor.execute('''
    INSERT OR IGNORE INTO LoggingAppFields (field_id, logging_app_id, app_transaction_id, app_transaction_id_description)
    VALUES 
        (1, 1, 'payment_id', 'Unique identifier for payment transactions'),
        (2, 2, 'customer_request_id', 'Unique identifier for customer requests'),
        (3, 3, 'inventory_transaction_id', 'Unique identifier for inventory movements')
    ''')
    
    cursor.execute('''
    INSERT OR IGNORE INTO AppEnrichmentAPI (app_api_id, logging_app_id, endpoint_name, endpoint_url, api_type, request_schema, response_schema)
    VALUES 
        (1, 1, 'Payment Enrichment', 'https://api.example.com/payments', 'rest', '{"transaction_id": "string"}', '{"transaction_id": "string", "customer_id": "string", "amount": "number", "status": "string"}'),
        (2, 2, 'Customer Enrichment', 'https://api.example.com/customers', 'rest', '{"request_id": "string"}', '{"request_id": "string", "customer_id": "string", "request_type": "string", "status": "string"}'),
        (3, 3, 'Inventory Enrichment', 'https://api.example.com/inventory', 'rest', '{"transaction_id": "string"}', '{"transaction_id": "string", "product_id": "string", "quantity": "number", "location": "string"}')
    ''')
    
    conn.commit()
    conn.close()

# Main Streamlit app
def main():
    st.set_page_config(page_title="Log Advisor", page_icon="📊", layout="wide")
    
    st.title("Log Advisor")
    st.subheader("Analyze application logs without exposing sensitive information")
    
    # Initialize database if needed
    if not os.path.exists('log_advisor.db'):
        init_database()
    
    # Initialize application service
    app_service = ApplicationService()
    
    try:
        # Get applications from database
        applications = app_service.get_applications()
        app_options = {app['app_name']: app['app_id'] for app in applications}
        
        # UI Form
        with st.form("log_analysis_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                app_name = st.selectbox("Application", options=list(app_options.keys()))
                transaction_id = st.text_input("Transaction ID")
            
            with col2:
                start_time = st.datetime_input("Transaction Start Time")
                end_time = st.datetime_input("Transaction End Time")
            
            submit_button = st.form_submit_button("Analyze Logs")
        
        # Process form submission
        if submit_button:
            if not transaction_id:
                st.error("Please enter a Transaction ID")
            else:
                app_id = app_options[app_name]
                start_time_str = start_time.isoformat()
                end_time_str = end_time.isoformat()
                
                # Show loading spinner
                with st.spinner("Analyzing logs..."):
                    # Process transaction
                    result = app_service.process_transaction(app_id, transaction_id, start_time_str, end_time_str)
                
                # Display results
                if "error" in result:
                    st.error(result["error"])
                else:
                    # Success - display results in tabs
                    tab1, tab2, tab3 = st.tabs(["Analysis", "Transaction Details", "Raw Logs"])
                    
                    with tab1:
                        st.subheader("Log Analysis")
                        st.markdown(result["analysis"])
                    
                    with tab2:
                        st.subheader("Transaction Details")
                        # Display transaction data without sensitive fields
                        trans_df = pd.DataFrame([result["transaction_data"]])
                        st.dataframe(trans_df)
                    
                    with tab3:
                        st.subheader("Raw Logs")
                        # Display logs in a table
                        logs_df = pd.DataFrame(result["logs"])
                        st.dataframe(logs_df)
    
    finally:
        # Close connections
        app_service.close()

if __name__ == "__main__":
    main()
