import pandas as pd
import oracledb
import os
from typing import List, Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def initialize_oracle_client(oracle_client_path: str = None):
    """
    Initialize the Oracle client if needed.
    
    Args:
        oracle_client_path: Path to Oracle client libraries
    """
    if oracle_client_path:
        try:
            oracledb.init_oracle_client(lib_dir=oracle_client_path)
            logger.info("Oracle client initialized successfully")
        except Exception as e:
            logger.warning(f"Could not initialize Oracle client: {e}")
            logger.info("Continuing without explicit initialization...")

def connect_to_oracle(username: str, password: str, dsn: str) -> oracledb.Connection:
    """
    Establish connection to Oracle database.
    
    Args:
        username: Oracle database username
        password: Oracle database password
        dsn: Oracle connection string
        
    Returns:
        Oracle connection object
    """
    try:
        connection = oracledb.connect(user=username, password=password, dsn=dsn)
        logger.info(f"Successfully connected to Oracle Database {dsn}")
        return connection
    except oracledb.Error as e:
        logger.error(f"Oracle Error: {e}")
        raise

def extract_project_ids(project_id_str: str) -> List[str]:
    """
    Extract project IDs from a comma-separated string.
    
    Args:
        project_id_str: String containing one or more project IDs
        
    Returns:
        List of project IDs
    """
    if pd.isna(project_id_str):
        return []
    
    # Convert to string in case it's a numeric value
    project_id_str = str(project_id_str).strip()
    
    if not project_id_str:
        return []
    
    # Split by comma and strip whitespace
    return [pid.strip() for pid in project_id_str.split(',') if pid.strip()]

def get_workflow_data(connection: oracledb.Connection, project_id: str) -> Dict[str, Any]:
    """
    Execute the custom query to get workflow data for a project ID.
    Replace the placeholder query with your actual query.
    
    Args:
        connection: Oracle connection object
        project_id: Project ID to query
        
    Returns:
        Dictionary containing workflow data
    """
    cursor = None
    try:
        cursor = connection.cursor()
        
        # Replace this with your actual query
        query = """
        SELECT 
            p.project_id,
            p.project_name,
            p.project_status,
            w.workflow_id,
            w.workflow_name,
            w.workflow_status,
            w.created_date,
            w.modified_date
        FROM 
            projects p
        JOIN 
            workflows w ON p.project_id = w.project_id
        WHERE 
            p.project_id = :project_id
        """
        
        cursor.execute(query, project_id=project_id)
        columns = [col[0] for col in cursor.description]
        rows = cursor.fetchall()
        
        if not rows:
            logger.warning(f"No data found for project ID: {project_id}")
            return None
        
        # Convert the first row to a dictionary
        result = dict(zip(columns, rows[0]))
        logger.info(f"Successfully retrieved data for project ID: {project_id}")
        return result
    
    except oracledb.Error as e:
        logger.error(f"Oracle Error for project ID {project_id}: {e}")
        return None
    finally:
        if cursor:
            cursor.close()

def process_excel_file(
    input_file: str, 
    output_file: str, 
    project_id_column: str,
    db_username: str, 
    db_password: str, 
    db_dsn: str,
    oracle_client_path: str = None,
    thick_mode: bool = False
):
    """
    Main function to process the Excel file, query Oracle, and write results.
    
    Args:
        input_file: Path to input Excel file
        output_file: Path to output Excel file
        project_id_column: Column name containing project IDs
        db_username: Oracle database username
        db_password: Oracle database password
        db_dsn: Oracle connection string
        oracle_client_path: Path to Oracle client libraries (optional)
        thick_mode: Whether to use thick mode (Oracle client) or thin mode (Python only)
    """
    try:
        # Set the client mode (thick or thin)
        if thick_mode:
            oracledb.defaults.config_dir = os.environ.get("TNS_ADMIN", "")
            # Initialize Oracle client if path provided
            initialize_oracle_client(oracle_client_path)
        else:
            # Use the thin mode (pure Python, no Oracle client needed)
            oracledb.init_mode = oracledb.INIT_MODE_THIN
            logger.info("Using thin mode (pure Python) for Oracle connection")
        
        # Read the Excel file
        logger.info(f"Reading input Excel file: {input_file}")
        df = pd.read_excel(input_file)
        
        if project_id_column not in df.columns:
            logger.error(f"Project ID column '{project_id_column}' not found in Excel file")
            return
        
        # Connect to Oracle database
        connection = connect_to_oracle(db_username, db_password, db_dsn)
        
        # Create a list to store all aggregated results
        all_results = []
        
        # Process each row in the Excel file
        for idx, row in df.iterrows():
            project_id_value = row[project_id_column]
            project_ids = extract_project_ids(project_id_value)
            
            if not project_ids:
                logger.warning(f"No valid project IDs found in row {idx+1}")
                continue
            
            logger.info(f"Processing row {idx+1} with project IDs: {project_ids}")
            
            # Process each project ID
            for project_id in project_ids:
                # Skip if not numeric
                if not project_id.isdigit():
                    logger.warning(f"Skipping non-numeric project ID: {project_id}")
                    continue
                
                # Get workflow data for this project ID
                workflow_data = get_workflow_data(connection, project_id)
                
                if workflow_data:
                    # Add original row data if needed
                    workflow_data['original_row_index'] = idx
                    all_results.append(workflow_data)
        
        # Close the database connection
        connection.close()
        
        if not all_results:
            logger.warning("No results found for any project ID")
            return
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(all_results)
        
        # Write results to Excel
        logger.info(f"Writing {len(results_df)} results to {output_file}")
        results_df.to_excel(output_file, index=False)
        logger.info(f"Successfully wrote results to {output_file}")
        
    except Exception as e:
        logger.error(f"Error processing Excel file: {e}")
        raise

if __name__ == "__main__":
    # Configuration parameters - replace with your actual values
    INPUT_FILE = "your_input_file.xlsx"
    OUTPUT_FILE = "ProjectDetails.xlsx"
    PROJECT_ID_COLUMN = "Project Id"  # Adjust column name as needed
    
    # Database connection parameters
    DB_USERNAME = "your_username"
    DB_PASSWORD = "your_password"
    DB_DSN = "your_dsn"  # Format: "host:port/service_name"
    
    # Optional: Path to Oracle client libraries for thick mode
    ORACLE_CLIENT_PATH = None  # e.g., "/path/to/instantclient"
    
    # Whether to use thick mode (Oracle client) or thin mode (Python only)
    # Thin mode is recommended for simplicity but thick mode offers more features
    THICK_MODE = False
    
    # Process the Excel file
    process_excel_file(
        input_file=INPUT_FILE,
        output_file=OUTPUT_FILE,
        project_id_column=PROJECT_ID_COLUMN,
        db_username=DB_USERNAME,
        db_password=DB_PASSWORD,
        db_dsn=DB_DSN,
        oracle_client_path=ORACLE_CLIENT_PATH,
        thick_mode=THICK_MODE
    )
