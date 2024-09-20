import psycopg2
import pandas as pd
from elasticsearch import Elasticsearch

# Step 1: Database connection and query execution
def connect_to_db(host, dbname, user, password):
    """Establish a connection to the PostgreSQL database."""
    connection = psycopg2.connect(
        host=host,
        dbname=dbname,
        user=user,
        password=password
    )
    return connection

def get_open_tickets(connection):
    """Fetch incident descriptions and incident numbers from ays_intake_view table."""
    query = "SELECT incident_number, description FROM ays_intake_view WHERE state NOT IN ('Closed', 'Open')"
    df = pd.read_sql_query(query, connection)
    return df

# Step 2: Extract order numbers from ticket descriptions
def extract_order_numbers(description):
    """Extract numeric order numbers from the description. Handle multiple numbers."""
    import re
    return re.findall(r'\d+', description)  # A simple regex to extract numeric parts

# Step 3: Match order number with t_raw_order table
def match_order_number(connection, possible_order_numbers):
    """Match order numbers from description with work_order_number in t_raw_order."""
    matched_order = None
    for order_num in possible_order_numbers:
        query = f"SELECT work_order_number, ORDER_SOURCE, ORDER_ACTION FROM t_raw_order WHERE work_order_number = {order_num}"
        df = pd.read_sql_query(query, connection)
        if not df.empty:
            matched_order = df.iloc[0]
            break
    if matched_order is None:
        return "t_raw_order has no information", None, None
    return f"Matched order: {matched_order['work_order_number']}", matched_order['ORDER_SOURCE'], matched_order['ORDER_ACTION']

# Step 4: Query t_master_order table to fetch additional details
def fetch_master_order_details(connection, order_number):
    """Fetch order details like LOB, STATUS, etc. from t_master_order."""
    query = f"SELECT * FROM t_master_order WHERE order_number = {order_number}"
    df = pd.read_sql_query(query, connection)
    if df.empty:
        return "t_master_order has no matching Order_number", None
    return df.iloc[0]

# Step 5: Query t_transaction_manager for work step details
def fetch_transaction_steps(connection, order_number):
    """Fetch work step details from t_transaction_manager."""
    query = f"SELECT * FROM t_transaction_manager WHERE order_number = {order_number}"
    df = pd.read_sql_query(query, connection)
    return df

# Step 6: Analyze work steps and order completion
def analyze_work_steps(order_status, transaction_steps):
    """Determine if the order is complete or find the failed step."""
    if order_status == 'WO_COMPLETE' and all(transaction_steps['state'] == 'SUCCESS'):
        return "Order is marked Completed"
    else:
        failed_steps = transaction_steps[transaction_steps['state'] != 'SUCCESS']
        if not failed_steps.empty:
            return f"Latest failed step: {failed_steps.iloc[-1]['work_step_name']}"
        return "No failed steps found"

# Step 7: Grep ELK logs for failed transactions
def search_elk_logs(elk_host, start_time, end_time, work_step_identifier):
    """Search ELK logs for errors related to a work step between the given start and end time."""
    es = Elasticsearch([elk_host])
    query = {
        "query": {
            "bool": {
                "must": [
                    {"match": {"work_step_identifier": work_step_identifier}},
                    {"range": {"@timestamp": {"gte": start_time, "lte": end_time}}}
                ]
            }
        }
    }
    res = es.search(index="logs-*", body=query)
    if res['hits']['hits']:
        return res['hits']['hits'][0]['_source']['message']
    return "No logs found for the failed work step"

# Step 8: Write analysis outcome into PostgreSQL table
def write_analysis_output(connection, analysis_df, table_name="ticket_analysis_output"):
    """Write the final analysis dataframe into a PostgreSQL table."""
    analysis_df.to_sql(table_name, con=connection, if_exists='replace', index=False)

# Main Function to Run the Analysis
def ticket_analyzer(connection, elk_host):
    """Main function to orchestrate the analysis process."""
    tickets_df = get_open_tickets(connection)
    analysis_data = []

    for _, row in tickets_df.iterrows():
        incident_number = row['incident_number']
        description = row['description']

        # Step 2: Extract order numbers from the description
        order_numbers = extract_order_numbers(description)

        # Step 3: Match order numbers in t_raw_order
        analysis, order_source, order_action = match_order_number(connection, order_numbers)
        
        if order_source is None:
            analysis_data.append([incident_number, description, analysis])
            continue

        # Step 4: Fetch order details from t_master_order
        master_order = fetch_master_order_details(connection, order_numbers[0])
        if master_order is None:
            analysis += "\nMaster order not found"
            analysis_data.append([incident_number, description, analysis])
            continue

        # Step 5: Fetch work steps from t_transaction_manager
        transaction_steps = fetch_transaction_steps(connection, order_numbers[0])

        # Step 6: Analyze work steps and determine completion status
        order_status = master_order['STATUS']
        analysis += "\n" + analyze_work_steps(order_status, transaction_steps)

        # Step 7: If there is a failed step, search ELK logs for the failure
        failed_step = transaction_steps[transaction_steps['state'] != 'SUCCESS']
        if not failed_step.empty:
            start_time = failed_step.iloc[0]['start_time']
            end_time = failed_step.iloc[0]['end_time']
            work_step_id = failed_step.iloc[0]['work_step_identifier']
            elk_log = search_elk_logs(elk_host, start_time, end_time, work_step_id)
            analysis += "\n" + elk_log

        # Collect the final analysis
        analysis_data.append([incident_number, description, analysis])

    # Create a pandas DataFrame and write the final analysis into the database
    analysis_df = pd.DataFrame(analysis_data, columns=['incident_number', 'description', 'analysis'])
    write_analysis_output(connection, analysis_df)

# Example Usage
if __name__ == "__main__":
    db_connection = connect_to_db('host1', 'db1', 'user', 'password')
    elk_host = 'http://elk-server-url'

    ticket_analyzer(db_connection, elk_host)
