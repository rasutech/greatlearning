import cx_Oracle  # Install with 'pip install cx_Oracle'
import psycopg2  # Install with 'pip install psycopg2'
from datetime import datetime, timedelta

# Database Connection Details (Replace with your credentials)
oracle_dsn = cx_Oracle.makedsn('your_oracle_hostname', 'port', service_name='your_service_name')
oracle_conn = cx_Oracle.connect('username', 'password', dsn=oracle_dsn)
postgres_conn = psycopg2.connect(database="your_database_name", user="your_username", password="your_password", host="your_host", port="5432")

def sync_data():
    with oracle_conn.cursor() as oracle_cur, postgres_conn.cursor() as pg_cur:
        # Calculate the 2-day window
        today = datetime.today()
        two_days_ago = today - timedelta(days=2)

        # Get modified/created tasks from Oracle within the 2-day window
        oracle_query = """
            SELECT * 
            FROM Fallout_tasking_table
            WHERE (MODIFIED_DATE >= :start_date AND MODIFIED_DATE < :end_date)
               OR (CREATED_DATE >= :start_date AND CREATED_DATE < :end_date)
        """
        oracle_cur.execute(oracle_query, start_date=two_days_ago, end_date=today)
        oracle_data = oracle_cur.fetchall()

        # Update/Insert into PostgreSQL
        for row in oracle_data:
            task_name = row
