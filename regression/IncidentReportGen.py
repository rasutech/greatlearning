import configparser
import pandas as pd
import psycopg2
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def load_config(config_file='config.ini'):
    """Load configuration from ini file"""
    config = configparser.ConfigParser()
    config.read(config_file)
    return config

def connect_to_db(db_config):
    """Create database connection"""
    return psycopg2.connect(
        host=db_config['host'],
        database=db_config['database'],
        user=db_config['user'],
        password=db_config['password']
    )

def fetch_incident_data(conn):
    """Fetch all incident data from PostgreSQL"""
    query = """
        SELECT 
            incident_number, app_id, state, impact, priority, urgency,
            created_on, assigned_to, assignment_group, opened_by,
            resolved_by, resolved_at, resolution_code
        FROM incidents
    """
    return pd.read_sql_query(query, conn)

def monthly_inflow_chart(df):
    """Generate monthly ticket inflow visualization"""
    monthly_counts = df.groupby([
        pd.Grouper(key='created_on', freq='M'),
        'app_id'
    ]).size().reset_index(name='count')
    
    fig = px.line(monthly_counts, 
                  x='created_on', 
                  y='count',
                  color='app_id',
                  title='Monthly Ticket Inflow by Application')
    return fig.to_html(full_html=False)

def top_5_apps_monthly(df):
    """Generate top 5 applications charts for each month"""
    charts = []
    
    for month in pd.date_range(df['created_on'].min(), df['created_on'].max(), freq='M'):
        month_data = df[df['created_on'].dt.to_period('M') == month.to_period('M')]
        top_5_apps = month_data['app_id'].value_counts().nlargest(5).index
        
        filtered_data = month_data[month_data['app_id'].isin(top_5_apps)]
        
        fig = px.bar(
            filtered_data['app_id'].value_counts().reset_index(),
            x='index',
            y='app_id',
            title=f'Top 5 Applications - {month.strftime("%B %Y")}',
            labels={'index': 'Application', 'app_id': 'Number of Tickets'}
        )
        charts.append(fig.to_html(full_html=False))
    
    return charts

def resolution_code_trends(df):
    """Generate monthly resolution code visualization"""
    monthly_resolution = df.groupby([
        pd.Grouper(key='resolved_at', freq='M'),
        'resolution_code'
    ]).size().reset_index(name='count')
    
    fig = px.area(monthly_resolution,
                  x='resolved_at',
                  y='count',
                  color='resolution_code',
                  title='Monthly Resolution Code Distribution')
    return fig.to_html(full_html=False)

def closure_trends(df):
    """Generate monthly closure trends visualization"""
    monthly_closure = df.groupby(pd.Grouper(key='resolved_at', freq='M')).size()
    monthly_inflow = df.groupby(pd.Grouper(key='created_on', freq='M')).size()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=monthly_inflow.index, y=monthly_inflow.values,
                            name='New Tickets', mode='lines+markers'))
    fig.add_trace(go.Scatter(x=monthly_closure.index, y=monthly_closure.values,
                            name='Closed Tickets', mode='lines+markers'))
    
    fig.update_layout(title='Monthly Ticket Inflow vs Closure Trends',
                     xaxis_title='Month',
                     yaxis_title='Number of Tickets')
    return fig.to_html(full_html=False)

def generate_html_report(charts, top_5_charts, config):
    """Generate final HTML report"""
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Incident Analysis Report</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .chart-container {{ margin-bottom: 40px; }}
            h1, h2 {{ color: #333; }}
        </style>
    </head>
    <body>
        <h1>Incident Analysis Report</h1>
        <div class="chart-container">
            <h2>Monthly Ticket Inflow</h2>
            {charts['monthly_inflow']}
        </div>
        
        <div class="chart-container">
            <h2>Top 5 Applications by Month</h2>
            {''.join(f'<div class="chart-container">{chart}</div>' for chart in top_5_charts)}
        </div>
        
        <div class="chart-container">
            <h2>Resolution Code Trends</h2>
            {charts['resolution_trends']}
        </div>
        
        <div class="chart-container">
            <h2>Closure Trends</h2>
            {charts['closure_trends']}
        </div>
    </body>
    </html>
    """
    
    with open(config['output']['html_path'], 'w') as f:
        f.write(html_template)

def main():
    # Load configuration
    config = load_config()
    
    # Connect to database
    conn = connect_to_db(config['database'])
    
    try:
        # Fetch data
        df = fetch_incident_data(conn)
        
        # Convert datetime columns
        df['created_on'] = pd.to_datetime(df['created_on'])
        df['resolved_at'] = pd.to_datetime(df['resolved_at'])
        
        # Generate charts
        charts = {
            'monthly_inflow': monthly_inflow_chart(df),
            'resolution_trends': resolution_code_trends(df),
            'closure_trends': closure_trends(df)
        }
        
        top_5_charts = top_5_apps_monthly(df)
        
        # Generate HTML report
        generate_html_report(charts, top_5_charts, config)
        
        print(f"Report generated successfully at {config['output']['html_path']}")
        
    finally:
        conn.close()

if __name__ == "__main__":
    main()
