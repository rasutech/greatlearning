import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime

# PostgreSQL connection details
db_url = 'postgresql://<username>:<password>@<hostname>:<port>/<database>'  # Update with your credentials
engine = create_engine(db_url)

# SQL query to fetch relevant data
query = """
SELECT 
    incident_number, 
    application_name, 
    resolved_by, 
    created_on, 
    resolved_at, 
    assignment_group, 
    urgency, 
    priority, 
    severity, 
    state 
FROM 
    myticks_busted 
WHERE 
    state IN ('resolved', 'closed') 
    AND created_on >= '2024-04-01' 
"""

# Read data into a DataFrame
df = pd.read_sql_query(query, engine)

# Data preprocessing
df['created_on'] = pd.to_datetime(df['created_on'])
df['resolved_at'] = pd.to_datetime(df['resolved_at'])
df['resolution_time'] = df['resolved_at'] - df['created_on']

# 1. Applications generating maximum tickets on a daily basis
daily_ticket_counts = df.groupby([pd.Grouper(key='created_on', freq='D'), 'application_name']).size().reset_index(name='ticket_count')
top_applications_daily = daily_ticket_counts.sort_values(['created_on', 'ticket_count'], ascending=[True, False]).groupby('created_on').head(1)

# 2. Top resolvers in each application
top_resolvers_per_app = df.groupby(['application_name', 'resolved_by']).size().reset_index(name='ticket_count')
top_resolvers_per_app = top_resolvers_per_app.sort_values(['application_name', 'ticket_count'], ascending=[True, False]).groupby('application_name').head(1)

# 3. Mean time to resolve by urgency for top-10 applications
top_10_apps = daily_ticket_counts.groupby('application_name')['ticket_count'].sum().nlargest(10).index
mean_resolution_time_by_urgency = df[df['application_name'].isin(top_10_apps)].groupby(['application_name', 'urgency'])['resolution_time'].mean()

# 4. Identifying non-productive and productive resolvers
tickets_resolved_per_day = df.groupby(['resolved_by', pd.Grouper(key='resolved_at', freq='D')]).size().reset_index(name='tickets_resolved')
mean_resolution_time_per_resolver = df.groupby('resolved_by')['resolution_time'].mean()

# Combine metrics to identify non-productive resolvers (low volume, high resolution time)
resolver_performance = pd.merge(tickets_resolved_per_day, mean_resolution_time_per_resolver, on='resolved_by')
non_productive_resolvers = resolver_performance[
    (resolver_performance['tickets_resolved'] < resolver_performance['tickets_resolved'].quantile(0.25)) & 
    (resolver_performance['resolution_time'] > resolver_performance['resolution_time'].quantile(0.75))
]

# Display results
print("1. Applications generating maximum tickets on a daily basis:")
print(top_applications_daily)

print("\n2. Top resolvers in each application:")
print(top_resolvers_per_app)

print("\n3. Mean time to resolve by urgency for top-10 applications:")
print(mean_resolution_time_by_urgency)

print("\n4. Non-productive resolvers (low volume, high resolution time):")
print(non_productive_resolvers)
