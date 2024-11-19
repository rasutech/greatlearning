import configparser
import pandas as pd
import psycopg2
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def calculate_working_days():
    """Calculate working days for each month excluding weekends and holidays"""
    # Define the date range
    start_date = pd.Timestamp('2023-11-01')
    end_date = pd.Timestamp('2024-10-31')
    
    # Define national holidays for the period
    holidays = pd.to_datetime([
        '2023-11-23',  # Thanksgiving
        '2023-12-25',  # Christmas
        '2024-01-01',  # New Year's Day
        '2024-01-15',  # Martin Luther King Jr. Day
        '2024-02-19',  # Presidents' Day
        '2024-05-27',  # Memorial Day
        '2024-07-04',  # Independence Day
        '2024-09-02',  # Labor Day
    ])
    
    # Generate date range
    date_range = pd.date_range(start=start_date, end=end_date)
    
    # Create DataFrame with dates
    working_days_df = pd.DataFrame({
        'date': date_range,
        'month': date_range.strftime('%Y-%m'),
        'is_weekend': date_range.weekday.isin([5, 6]),  # 5=Saturday, 6=Sunday
        'is_holiday': date_range.isin(holidays)
    })
    
    # Calculate working days for each month
    monthly_working_days = working_days_df.groupby('month').apply(
        lambda x: (~(x['is_weekend'] | x['is_holiday'])).sum()
    ).reset_index()
    monthly_working_days.columns = ['month', 'working_days']
    
    return monthly_working_days

def create_working_days_correlation(df):
    """Create visualization showing correlation between working days and ticket volume"""
    # Calculate working days
    working_days_df = calculate_working_days()
    
    # Calculate monthly ticket counts
    monthly_tickets = df.groupby(
        df['created_on'].dt.strftime('%Y-%m')
    ).size().reset_index()
    monthly_tickets.columns = ['month', 'ticket_count']
    
    # Merge working days with ticket counts
    correlation_data = pd.merge(working_days_df, monthly_tickets, on='month')
    
    # Calculate correlation coefficient
    correlation = correlation_data['working_days'].corr(correlation_data['ticket_count'])
    
    # Create scatter plot
    fig = go.Figure()
    
    # Add scatter plot
    fig.add_trace(go.Scatter(
        x=correlation_data['working_days'],
        y=correlation_data['ticket_count'],
        mode='markers+text',
        text=correlation_data['month'],
        textposition="top center",
        marker=dict(
            size=10,
            color='rgb(55, 83, 109)',
            line=dict(width=1)
        ),
        name='Monthly Data'
    ))
    
    # Add trend line
    z = np.polyfit(correlation_data['working_days'], correlation_data['ticket_count'], 1)
    p = np.poly1d(z)
    fig.add_trace(go.Scatter(
        x=correlation_data['working_days'],
        y=p(correlation_data['working_days']),
        mode='lines',
        name='Trend Line',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        title=f'Working Days vs Ticket Volume<br>Correlation Coefficient: {correlation:.3f}',
        xaxis_title='Number of Working Days',
        yaxis_title='Number of Tickets',
        showlegend=True,
        annotations=[
            dict(
                x=correlation_data['working_days'].min(),
                y=correlation_data['ticket_count'].max(),
                text=f"R² = {correlation**2:.3f}",
                showarrow=False,
                font=dict(size=12)
            )
        ]
    )
    
    # Add detailed analysis table
    analysis_table = go.Figure(data=[go.Table(
        header=dict(
            values=['Month', 'Working Days', 'Tickets', 'Tickets/Working Day'],
            fill_color='rgb(55, 83, 109)',
            font=dict(color='white'),
            align='left'
        ),
        cells=dict(
            values=[
                correlation_data['month'],
                correlation_data['working_days'],
                correlation_data['ticket_count'],
                (correlation_data['ticket_count'] / correlation_data['working_days']).round(2)
            ],
            fill_color='white',
            align='left'
        )
    )])
    
    analysis_table.update_layout(
        title='Monthly Analysis of Tickets per Working Day',
        margin=dict(t=30, b=10)
    )
    
    return fig.to_html(full_html=False), analysis_table.to_html(full_html=False)

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

# Update the generate_html_report function to include the new visualization
def generate_html_report(charts, top_10_charts, top_10_distribution_charts, working_days_charts, config):
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
            <h2>Working Days Analysis</h2>
            {working_days_charts[0]}
            <div class="summary">
                <h3>Detailed Working Days Analysis</h3>
                {working_days_charts[1]}
            </div>
        </div>
        
        <div class="chart-container">
            <h2>Top 10 Applications Distribution by Month</h2>
            <div class="chart-grid">
                {''.join(f'<div>{chart}</div>' for chart in top_10_distribution_charts)}
            </div>
        </div>
        
        <div class="chart-container">
            <h2>Top 10 Applications Details by Month</h2>
            <div class="chart-grid">
                {''.join(f'<div>{chart}</div>' for chart in top_10_charts)}
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
        with codecs.open(config['output']['html_path'], 'w', encoding='utf-8') as f:
            f.write(html_template)
    except Exception as e:
        try:
            with open(config['output']['html_path'], 'w', encoding='utf-8', errors='replace') as f:
                f.write(html_template)
        except Exception as e:
            cleaned_template = html_template.encode('ascii', 'ignore').decode('ascii')
            with open(config['output']['html_path'], 'w', encoding='utf-8') as f:
                f.write(cleaned_template)
            print("Warning: Some special characters were removed from the report due to encoding issues.")

# Update the main function to include the new visualization
def main():
    config = load_config()
    
    if 'database' not in config or 'output' not in config:
        raise ValueError("Configuration file is missing required sections")
    
    try:
        conn = connect_to_db(config['database'])
        df = fetch_incident_data(conn)
        
        try:
            df['created_on'] = pd.to_datetime(df['created_on'])
            df['resolved_at'] = pd.to_datetime(df['resolved_at'])
        except Exception as e:
            print(f"Error converting datetime columns: {e}")
            raise
        
        try:
            charts = {
                'monthly_inflow': monthly_inflow_chart(df),
                'resolution_trends': resolution_code_trends(df),
                'closure_trends': closure_trends(df)
            }
            
            # Generate Top-10 related charts
            top_10_charts = top_10_apps_monthly(df)
            top_10_distribution_charts = top_10_distribution_monthly(df)
            
            # Generate working days correlation analysis
            working_days_charts = create_working_days_correlation(df)
            
        except Exception as e:
            print(f"Error generating charts: {e}")
            raise
        
        try:
            generate_html_report(charts, top_10_charts, top_10_distribution_charts, working_days_charts, config)
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

if __name__ == "__main__":
    main()
