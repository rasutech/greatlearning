import requests
import psycopg2
import yaml  # or json if your OpenAPI spec is in JSON format
from openapi_schema_validator import validate

# Load your OpenAPI specification
with open("your_swagger_file.yaml", "r") as f:
    spec = yaml.safe_load(f)

# Database configuration (same as before)
db_config = {...}

# API configuration
api_base_url = "https://your-api-endpoint"

# Fetch failed work orders
with psycopg2.connect(**db_config) as conn:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT work_order, task_name, status, transaction_time
            FROM work_orders
            WHERE status = 'FAILED';
        """)
        failed_orders = cur.fetchall()

# Process each failed work order
for order in failed_orders:
    work_order, _, _, _ = order

    # Fetch location_id based on work_order
    with conn.cursor() as cur:
        cur.execute("SELECT location_id FROM locations WHERE work_order = %s;", (work_order,))
        location_id = cur.fetchone()[0]
    
    # Determine eventType based on work_order_type
    # ... (Your logic to fetch work_order_type and decide eventType)

    # Call the REST API
    try:
        # Find the operation in the spec based on path and method
        rest_api_path = "/your_rest_api_endpoint"  
        rest_api_method = "get" 
        rest_api_operation = next(op for path, methods in spec["paths"].items() 
                                    if path == rest_api_path 
                                    for method, op in methods.items() 
                                    if method == rest_api_method)

        # Construct the URL and parameters
        url = api_base_url + rest_api_path
        params = {param["name"]: value for param in rest_api_operation.get("parameters", []) 
                    if param["in"] == "query" and param["name"] in ["location_id", "event_type"]}

        response = requests.get(
            url,
            params=params,
            auth=("your_swagger_username", "your_swagger_password")
        )
        response.raise_for_status()

        # Validate the response
        validate(response.json(), rest_api_operation["responses"]["200"]["content"]["application/json"]["schema"])

        # ... (Handle the response)

    except requests.exceptions.RequestException as e:
        print(f"Error calling the REST API: {e}")
    
    # Extract process_id and container_id from transaction_id
    # ... (Your logic to parse transaction_id)

    # Call the second API (PUT request - similar approach using spec)
    # ... (Use the OpenAPI spec to determine URL, parameters, payload schema)
