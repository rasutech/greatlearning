import configparser
import pandas as pd
import psycopg2
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import codecs

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
    
    # Convert created_on to datetime if it's not already
    df['created_on'] = pd.to_datetime(df['created_on'])
    
    # Get unique months in the data
    months = df['created_on'].dt.to_period('M').unique()
    
    for month in months:
        # Filter data for the current month
        month_data = df[df['created_on'].dt.to_period('M') == month]
        
        # Get top 5 apps for this month
        app_counts = month_data['app_id'].value_counts().head(5)
        
        # Create DataFrame for plotting
        plot_data = pd.DataFrame({
            'Application': app_counts.index,
            'Tickets': app_counts.values
        })
        
        # Create the bar chart
        fig = px.bar(
            plot_data,
            x='Application',
            y='Tickets',
            title=f'Top 5 Applications - {month.strftime("%B %Y")}',
        )
        
        # Update layout
        fig.update_layout(
            xaxis_title='Application ID',
            yaxis_title='Number of Tickets',
            showlegend=False
        )
        
        charts.append(fig.to_html(full_html=False))
    
    return charts

def resolution_code_trends(df):
    """Generate monthly resolution code visualization"""
    # Ensure we only use rows where resolution_code is not null
    df_resolved = df.dropna(subset=['resolved_at', 'resolution_code'])
    
    monthly_resolution = df_resolved.groupby([
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
    # Ensure we only use valid dates
    df_resolved = df.dropna(subset=['resolved_at'])
    
    monthly_closure = df_resolved.groupby(
        pd.Grouper(key='resolved_at', freq='M')).size().reset_index(name='closed')
    monthly_inflow = df.groupby(
        pd.Grouper(key='created_on', freq='M')).size().reset_index(name='created')
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly_inflow['created_on'], 
        y=monthly_inflow['created'],
        name='New Tickets', 
        mode='lines+markers'
    ))
    fig.add_trace(go.Scatter(
        x=monthly_closure['resolved_at'], 
        y=monthly_closure['closed'],
        name='Closed Tickets', 
        mode='lines+markers'
    ))
    
    fig.update_layout(
        title='Monthly Ticket Inflow vs Closure Trends',
        xaxis_title='Month',
        yaxis_title='Number of Tickets'
    )
    return fig.to_html(full_html=False)

def top_5_distribution_monthly(df):
    """Generate monthly pie charts showing distribution of Top-5 apps vs Others"""
    charts = []
    
    # Convert created_on to datetime if it's not already
    df['created_on'] = pd.to_datetime(df['created_on'])
    
    # Get unique months in the data
    months = df['created_on'].dt.to_period('M').unique()
    
    for month in months:
        # Filter data for the current month
        month_data = df[df['created_on'].dt.to_period('M') == month]
        
        # Get total tickets for the month
        total_tickets = len(month_data)
        
        # Get top 5 apps and their counts
        top_5_apps = month_data['app_id'].value_counts().head(5)
        
        # Calculate others
        others_count = total_tickets - top_5_apps.sum()
        
        # Create plot data
        plot_data = pd.concat([
            top_5_apps,
            pd.Series({'Others': others_count})
        ])
        
        # Calculate percentages
        plot_data_pct = (plot_data / total_tickets * 100).round(1)
        
        # Create labels with percentages
        labels = [f'{idx} ({val}%)' for idx, val in plot_data_pct.items()]
        
        # Create the pie chart
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=plot_data.values,
            hole=0.3
        )])
        
        fig.update_layout(
            title=f'Ticket Distribution - {month.strftime("%B %Y")}<br>'
            f'(Top 5 Apps vs Others)<br>'
            f'Total Tickets: {total_tickets}',
            annotations=[{
                'text': f'{plot_data_pct["Others"]:.1f}% Others',
                'x': 0.5,
                'y': 0.5,
                'font_size': 12,
                'showarrow': False
            }]
        )
        
        charts.append(fig.to_html(full_html=False))
    
    return charts

def generate_html_report(charts, top_5_charts, top_5_distribution_charts, config):
    """Generate final HTML report with proper Unicode handling"""
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Incident Analysis Report</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .chart-container {{ margin-bottom: 40px; }}
            .chart-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(600px, 1fr));
                gap: 20px;
            }}
            h1, h2 {{ color: #333; }}
            .summary {{
                background-color: #f5f5f5;
                padding: 20px;
                border-radius: 5px;
                margin-bottom: 20px;
            }}
        </style>
    </head>
    <body>
        <h1>Incident Analysis Report</h1>
        
        <div class="chart-container">
            <h2>Monthly Ticket Inflow</h2>
            {charts['monthly_inflow']}
        </div>
        
        <div class="chart-container">
            <h2>Top 5 Applications Distribution by Month</h2>
            <div class="chart-grid">
                {''.join(f'<div>{chart}</div>' for chart in top_5_distribution_charts)}
            </div>
        </div>
        
        <div class="chart-container">
            <h2>Top 5 Applications Details by Month</h2>
            <div class="chart-grid">
                {''.join(f'<div>{chart}</div>' for chart in top_5_charts)}
            </div>
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
    
    try:
        # First attempt: Try writing with UTF-8 encoding using codecs
        with codecs.open(config['output']['html_path'], 'w', encoding='utf-8') as f:
            f.write(html_template)
    except Exception as e:
        try:
            # Second attempt: If that fails, try writing with error handling
            with open(config['output']['html_path'], 'w', encoding='utf-8', errors='replace') as f:
                f.write(html_template)
        except Exception as e:
            # Third attempt: If all else fails, try to remove problematic characters
            cleaned_template = html_template.encode('ascii', 'ignore').decode('ascii')
            with open(config['output']['html_path'], 'w', encoding='utf-8') as f:
                f.write(cleaned_template)
            print("Warning: Some special characters were removed from the report due to encoding issues.")

def main():
    # Load configuration
    config = load_config()
    
    # Add error handling for configuration
    if 'database' not in config or 'output' not in config:
        raise ValueError("Configuration file is missing required sections")
    
    try:
        # Connect to database with error handling
        conn = connect_to_db(config['database'])
        
        # Fetch data
        df = fetch_incident_data(conn)
        
        # Convert datetime columns with error handling
        try:
            df['created_on'] = pd.to_datetime(df['created_on'])
            df['resolved_at'] = pd.to_datetime(df['resolved_at'])
        except Exception as e:
            print(f"Error converting datetime columns: {e}")
            raise
        
        # Generate charts with error handling
        try:
            charts = {
                'monthly_inflow': monthly_inflow_chart(df),
                'resolution_trends': resolution_code_trends(df),
                'closure_trends': closure_trends(df)
            }
            
            # Generate Top-5 related charts
            top_5_charts = top_5_apps_monthly(df)
            top_5_distribution_charts = top_5_distribution_monthly(df)
            
        except Exception as e:
            print(f"Error generating charts: {e}")
            raise
        
        # Generate HTML report
        try:
            generate_html_report(charts, top_5_charts, top_5_distribution_charts, config)
            print(f"Report generated successfully at {config['output']['html_path']}")
        except Exception as e:
            print(f"Error generating HTML report: {e}")
            raise
            
    except Exception as e:
        print(f"An error occurred: {e}")
        raise
    
    finally:
        if 'conn' in locals() and conn is not None:
            conn.close()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Program terminated with error: {e}")
        exit(1)
