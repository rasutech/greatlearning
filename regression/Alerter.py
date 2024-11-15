import pandas as pd
import psycopg2
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from jinja2 import Template

def connect_to_db():
    """Establish database connection"""
    try:
        conn = psycopg2.connect(
            dbname="your_database",
            user="your_username",
            password="your_password",
            host="your_host",
            port="your_port"
        )
        return conn
    except Exception as e:
        raise Exception(f"Error connecting to database: {str(e)}")

def fetch_alerts_data(conn, app_name):
    """Fetch alerts data for specified app_name"""
    query = """
    SELECT incident_id, app_name, policy_name, condition_name, category,
           entity_type, entity_name, alert_description, is_root_incident,
           alert_start_time, priority, alert_end_time, datasource, environment
    FROM alerts_table
    WHERE app_name = %s
    AND alert_start_time >= NOW() - INTERVAL '30 days'
    """
    return pd.read_sql_query(query, conn, params=(app_name,))

def analyze_alert_patterns(df):
    """Analyze patterns in alerts data"""
    results = {}
    
    # Alert frequency by category and priority
    results['category_priority_counts'] = pd.crosstab(df['category'], df['priority'])
    
    # Average duration of alerts
    df['alert_duration'] = (df['alert_end_time'] - df['alert_start_time']).dt.total_seconds() / 3600
    results['avg_duration_by_category'] = df.groupby('category')['alert_duration'].mean()
    
    # Identify potential redundant alerts
    # Group by similar alerts within short time windows
    df['time_bucket'] = df['alert_start_time'].dt.round('10min')
    redundant_alerts = df.groupby(['time_bucket', 'condition_name']).size().reset_index(name='count')
    results['potential_redundant_alerts'] = redundant_alerts[redundant_alerts['count'] > 1]
    
    # Most common conditions
    results['top_conditions'] = df['condition_name'].value_counts().head(10)
    
    # Alert patterns by hour of day
    df['hour'] = df['alert_start_time'].dt.hour
    results['hourly_patterns'] = df.groupby('hour').size()
    
    return results

def create_visualizations(df, analysis_results):
    """Create plotly visualizations"""
    figs = []
    
    # Alert frequency by category and priority
    fig1 = px.bar(analysis_results['category_priority_counts'],
                  title='Alert Frequency by Category and Priority')
    figs.append(fig1)
    
    # Average duration by category
    fig2 = px.bar(analysis_results['avg_duration_by_category'],
                  title='Average Alert Duration by Category (Hours)')
    figs.append(fig2)
    
    # Hourly patterns
    fig3 = px.line(analysis_results['hourly_patterns'],
                   title='Alert Patterns by Hour of Day')
    figs.append(fig3)
    
    # Top conditions
    fig4 = px.bar(analysis_results['top_conditions'],
                  title='Most Common Alert Conditions')
    figs.append(fig4)
    
    return figs

def generate_html_report(app_name, df, analysis_results, figs):
    """Generate HTML report using Jinja2 template"""
    template_string = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Alert Analysis Report - {{ app_name }}</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .section { margin-bottom: 30px; }
            table { border-collapse: collapse; width: 100%; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
        </style>
    </head>
    <body>
        <h1>Alert Analysis Report - {{ app_name }}</h1>
        <p>Generated on: {{ generation_time }}</p>
        
        <div class="section">
            <h2>Dataset Overview</h2>
            <p>Total number of alerts: {{ total_alerts }}</p>
            <p>Date range: {{ date_range }}</p>
        </div>
        
        <div class="section">
            <h2>Potential Redundant Alerts</h2>
            {{ redundant_alerts_table }}
        </div>
        
        {% for fig in figures %}
        <div class="section">
            {{ fig }}
        </div>
        {% endfor %}
        
    </body>
    </html>
    """
    
    template = Template(template_string)
    
    # Convert plotly figures to HTML
    figure_htmls = [fig.to_html(full_html=False) for fig in figs]
    
    # Format redundant alerts table
    redundant_alerts_html = analysis_results['potential_redundant_alerts'].to_html()
    
    html_content = template.render(
        app_name=app_name,
        generation_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        total_alerts=len(df),
        date_range=f"{df['alert_start_time'].min()} to {df['alert_start_time'].max()}",
        redundant_alerts_table=redundant_alerts_html,
        figures=figure_htmls
    )
    
    # Save the HTML file
    filename = f"{app_name.lower()}_nov_15_2024.html"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return filename

def main(app_name):
    """Main function to run the analysis"""
    try:
        # Connect to database
        conn = connect_to_db()
        
        # Fetch data
        df = fetch_alerts_data(conn, app_name)
        
        if len(df) == 0:
            raise ValueError(f"No alerts found for app: {app_name}")
        
        # Perform analysis
        analysis_results = analyze_alert_patterns(df)
        
        # Create visualizations
        figs = create_visualizations(df, analysis_results)
        
        # Generate HTML report
        filename = generate_html_report(app_name, df, analysis_results, figs)
        
        print(f"Analysis complete. Report generated: {filename}")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    app_name = "YOUR_APP_NAME"  # Replace with your app name
    main(app_name)
