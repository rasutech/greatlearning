import ipywidgets as widgets
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
import seaborn as sns

def plot_ticket_counts_by_app(app_name):
    """Plot ticket counts for the selected application."""
    data_filtered = df[df['app_name'] == app_name]
    sns.countplot(data=data_filtered, x='priority')
    plt.title(f'Ticket Counts by Priority for {app_name}')
    plt.xticks(rotation=45)
    plt.ylabel('Ticket Count')

def plot_ticket_count_by_priority_and_month(app_name):
    """Plot ticket count by priority in the selected application across months."""
    data_filtered = df[df['app_name'] == app_name]
    sns.catplot(data=data_filtered, x='year_month', hue='priority', kind='count', height=4, aspect=2)
    plt.title(f'Ticket Count by Priority in {app_name} by Month')
    plt.xticks(rotation=45)
    plt.subplots_adjust(hspace=0.4)

# Creating a Dropdown menu for app_name selection
app_names = df['app_name'].unique()
dropdown_app_name = widgets.Dropdown(options=app_names, description='Select Application:')

output = widgets.Output()

# Widget to display visualizations based on the selected app_name
def update_plots(change):
    with output:
        clear_output(wait=True)  # Clear the output area before displaying the new plot
        plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
        plot_ticket_counts_by_app(change.new)
        plt.show()
        
        plt.figure(figsize=(14, 8))  # Adjust the figure size as needed for the second plot
        plot_ticket_count_by_priority_and_month(change.new)
        plt.show()

# Linking the dropdown to the update function
dropdown_app_name.observe(update_plots, names='value')

# Display the dropdown and output area
display(dropdown_app_name, output)


# Widget to display visualizations based on the selected app_id
def update_plots(change):
    plt.close('all')  # Close existing plots to prevent them from stacking
    plot_ticket_counts_by_app(change.new)
    plot_ticket_count_by_priority_and_month(change.new)

# Linking the dropdown to the update function
dropdown_app_id.observe(update_plots, names='value')

# Display the dropdown
display(dropdown_app_id)

from sqlalchemy import create_engine
import pandas as pd

# Database connection details
DATABASE = 'your_database'
USER = 'your_username'
PASSWORD = 'your_password'
HOST = 'localhost'
PORT = '5432'

# Create database connection
engine = create_engine(f'postgresql://{USER}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}')

# SQL query to fetch the data
sql_query = """
SELECT *, EXTRACT(YEAR FROM created_on) AS year, EXTRACT(MONTH FROM created_on) AS month
FROM your_table_name
WHERE created_on BETWEEN '2023-01-01' AND '2024-02-28'
"""

# Fetch data into DataFrame
df = pd.read_sql(sql_query, engine)

df['resolution_time'] = (df['resolved_on'] - df['created_on']).dt.total_seconds() / 3600  # Convert resolution time to hours
mttr = df.groupby(['app_id', 'state', 'impact', 'priority', 'urgency', 'active', 'assignment_group', 'year', 'month'])['resolution_time'].mean().reset_index()

import matplotlib.pyplot as plt
import seaborn as sns

# Ticket Counts per Application
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='app_id')
plt.title('Ticket Counts per Application')
plt.xticks(rotation=45)
plt.ylabel('Ticket Count')
plt.show()
