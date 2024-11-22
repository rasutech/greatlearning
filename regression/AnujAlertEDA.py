import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import warnings
warnings.filterwarnings('ignore')

def connect_to_db():
    """Create database connection"""
    # Replace these with your actual database credentials
    connection_string = "postgresql://username:password@host:port/dbname"
    return create_engine(connection_string)

def fetch_data(engine, app_name):
    """Fetch alerts data from PostgreSQL"""
    query = """
    SELECT *
    FROM alerts_table
    WHERE app_name = %s
    AND environment = 'production'
    AND priority = 'critical'
    """
    return pd.read_sql_query(query, engine, params=(app_name,))

def analyze_top_conditions(df):
    """Analyze top 10 conditions for each combination"""
    combinations = df.groupby(['datasource', 'category', 'policy_name'])
    
    plots = []
    for (ds, cat, policy), group in combinations:
        # Count conditions
        condition_counts = group['condition_name'].value_counts().head(10)
        
        # Create bar plot
        fig = px.bar(
            x=condition_counts.index,
            y=condition_counts.values,
            title=f'Top 10 Conditions for {ds} - {cat} - {policy}',
            labels={'x': 'Condition Name', 'y': 'Count'}
        )
        fig.update_layout(showlegend=False)
        plots.append(fig)
    
    return plots

def analyze_resolutions(df):
    """Analyze resolution types and statuses"""
    resolution_analysis = df.groupby(
        ['datasource', 'category', 'condition_name', 'policy_name']
    ).agg({
        'resolution_type': lambda x: x.value_counts().to_dict(),
        'resolution_status': lambda x: x.value_counts().to_dict()
    }).reset_index()
    
    return resolution_analysis

def analyze_alert_descriptions(df):
    """Perform text classification and grouping of alert descriptions"""
    results = []
    
    for (ds, cat), group in df.groupby(['datasource', 'category']):
        # Clean and get unique descriptions
        descriptions = group['alert_description'].fillna('')  # Replace None with empty string
        descriptions = descriptions[descriptions.str.len() > 0]  # Remove empty strings
        unique_descriptions = descriptions.unique()
        
        # Skip if we have too few descriptions
        if len(unique_descriptions) < 2:
            continue
            
        try:
            # Create TF-IDF vectors
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(unique_descriptions)
            
            # Calculate similarity matrix
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            # Find similar descriptions
            similar_groups = []
            
            for i, desc in enumerate(unique_descriptions):
                # Get indices of similar descriptions, excluding self-similarity
                similar_scores = list(enumerate(similarity_matrix[i]))
                # Sort by similarity score, exclude self (index i)
                similar_scores = sorted(
                    [item for item in similar_scores if item[0] != i],
                    key=lambda x: x[1],
                    reverse=True
                )
                
                # Get top 5 similar descriptions (or fewer if not enough available)
                num_similar = min(5, len(similar_scores))
                if num_similar > 0:
                    similar_indices = [score[0] for score in similar_scores[:num_similar]]
                    similar_desc = unique_descriptions[similar_indices].tolist()
                    
                    similar_groups.append({
                        'datasource': ds,
                        'category': cat,
                        'base_description': desc,
                        'similar_descriptions': similar_desc,
                        'similarity_scores': [f"{score[1]:.2f}" for score in similar_scores[:num_similar]]
                    })
            
            # Sort groups by average similarity score and take top 5
            if similar_groups:
                similar_groups.sort(
                    key=lambda x: sum(float(score) for score in x['similarity_scores']) / len(x['similarity_scores']),
                    reverse=True
                )
                results.extend(similar_groups[:5])
                
        except Exception as e:
            print(f"Warning: Error processing {ds}-{cat}: {str(e)}")
            continue
    
    # Convert results to DataFrame
    if not results:
        # Return empty DataFrame with correct columns if no results
        return pd.DataFrame(columns=[
            'datasource', 'category', 'base_description', 
            'similar_descriptions', 'similarity_scores'
        ])
    
    return pd.DataFrame(results)

def analyze_alert_age(df):
    """Calculate statistics for alert age"""
    # Convert timestamps and calculate duration
    df['alert_age'] = (pd.to_datetime(df['alert_end_time']) - 
                      pd.to_datetime(df['alert_start_time'])).dt.total_seconds() / 3600  # in hours
    
    stats = df.groupby(['datasource', 'condition_name', 'policy_name']).agg({
        'alert_age': ['mean', 'median', 'std']
    }).round(2)
    
    return stats

def additional_analysis(df):
    """Perform additional useful EDA"""
    additional_insights = {
        'hourly_distribution': px.histogram(
            df,
            x=pd.to_datetime(df['alert_start_time']).dt.hour,
            title='Alert Distribution by Hour of Day',
            labels={'x': 'Hour of Day', 'y': 'Count'}
        ),
        
        'weekly_pattern': px.histogram(
            df,
            x=pd.to_datetime(df['alert_start_time']).dt.day_name(),
            title='Alert Distribution by Day of Week',
            category_orders={'x': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']}
        ),
        
        'entity_distribution': px.pie(
            df['entity_type'].value_counts().reset_index(),
            values='count',
            names='entity_type',
            title='Distribution of Entity Types'
        ),
        
        'alert_duration_boxplot': px.box(
            df,
            x='category',
            y='alert_age',
            title='Alert Duration Distribution by Category'
        )
    }
    
    return additional_insights

def analyze_monthly_trends(df):
    """Analyze monthly trends for each datasource and category combination"""
    # Convert alert_start_time to datetime if it's not already
    df['alert_start_time'] = pd.to_datetime(df['alert_start_time'])
    
    # Extract month-year
    df['month_year'] = df['alert_start_time'].dt.strftime('%Y-%m')
    
    # Group by month, datasource, and category
    monthly_counts = df.groupby(['month_year', 'datasource', 'category']).size().reset_index(name='count')
    
    # Sort by month_year to ensure chronological order
    monthly_counts = monthly_counts.sort_values('month_year')
    
    # Create plots for each datasource
    trend_plots = []
    
    # Plot for each datasource
    for ds in df['datasource'].unique():
        ds_data = monthly_counts[monthly_counts['datasource'] == ds]
        
        fig = px.line(
            ds_data,
            x='month_year',
            y='count',
            color='category',
            title=f'Monthly Alert Trends for {ds}',
            labels={
                'month_year': 'Month',
                'count': 'Number of Alerts',
                'category': 'Category'
            }
        )
        
        # Customize layout
        fig.update_layout(
            xaxis_tickangle=-45,
            legend_title='Category',
            hovermode='x unified'
        )
        
        # Add markers to the lines
        fig.update_traces(mode='lines+markers')
        
        trend_plots.append(fig)
    
    # Create a summary table
    summary_table = monthly_counts.pivot_table(
        index=['month_year'],
        columns=['datasource', 'category'],
        values='count',
        fill_value=0
    ).round(2)
    
    return trend_plots, summary_table

def generate_html_report(app_name, plots, resolution_analysis, text_analysis, age_stats, additional_insights):
    """Generate HTML report with all analyses"""
    # Generate filename with sequence number
    date_str = datetime.now().strftime('%m_%d_%Y')
    base_filename = f'app_{date_str}'
    
    # Find next available sequence number
    seq = 1
    while os.path.exists(f'{base_filename}_{seq}.html'):
        seq += 1
    
    filename = f'{base_filename}_{seq}.html'
    
    # Create HTML content
    html_content = f"""
    <html>
    <head>
        <title>Alert Analysis Report - {app_name}</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .section {{ margin-bottom: 30px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <h1>Alert Analysis Report - {app_name}</h1>
        <div class="section">
            <h2>Top 10 Conditions Analysis</h2>
            {''.join([plot.to_html(full_html=False, include_plotlyjs=False) for plot in plots])}
        </div>
        
        <div class="section">
            <h2>Resolution Analysis</h2>
            {resolution_analysis.to_html()}
        </div>
        
        <div class="section">
            <h2>Similar Alert Descriptions</h2>
            {text_analysis.to_html()}
        </div>
        
        <div class="section">
            <h2>Alert Age Statistics (hours)</h2>
            {age_stats.to_html()}
        </div>
        
        <div class="section">
            <h2>Additional Insights</h2>
            {''.join([plot.to_html(full_html=False, include_plotlyjs=False) for plot in additional_insights.values()])}
        </div>
    </body>
    </html>
    """
    
    with open(filename, 'w') as f:
        f.write(html_content)
    
    return filename

def main(app_name='MYAPP'):
    # Connect to database and fetch data
    engine = connect_to_db()
    df = fetch_data(engine, app_name)
    
    # Perform analyses
    condition_plots = analyze_top_conditions(df)
    resolution_analysis = analyze_resolutions(df)
    text_analysis = analyze_alert_descriptions(df)
    age_stats = analyze_alert_age(df)
    additional_insights = additional_analysis(df)

    print("6. Analyzing monthly trends...")
    trend_plots, trend_summary = analyze_monthly_trends(df)
    
    # Generate report
    report_file = generate_html_report(
        app_name,
        condition_plots,
        resolution_analysis,
        text_analysis,
        age_stats,
        additional_insights,
        trend_plots,
        trend_summary
    )
    
    print(f"Analysis complete. Report generated: {report_file}")

if __name__ == "__main__":
    main()
