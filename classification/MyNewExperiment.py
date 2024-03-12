import ipywidgets as widgets
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
import seaborn as sns

# Unique values for app_name and assignment_group
app_names = df['app_name'].unique()
assignment_groups = df['assignment_group'].unique()
# Assuming 'priority' is a column in your DataFrame
priorities = df['priority'].unique()

# Create dropdown for priority
dropdown_priority = widgets.Dropdown(options=priorities, description='Select Priority:')

# Create dropdowns
dropdown_app_name = widgets.Dropdown(options=app_names, description='Select Application:')
dropdown_assignment_group = widgets.Dropdown(options=assignment_groups, description='Select Assignment Group:')

output = widgets.Output()

def update_assignment_group_dropdown(*args):
    app_name_selected = dropdown_app_name.value
    filtered_groups = df[df['app_name'] == app_name_selected]['assignment_group'].unique()
    dropdown_assignment_group.options = filtered_groups

# Call the update function whenever the app_name selection changes
dropdown_app_name.observe(update_assignment_group_dropdown, 'value')

def plot_data(app_name, assignment_group, priority):
    with output:
        clear_output(wait=True)
        # Filter data based on selections
        filtered_df = df[(df['app_name'] == app_name) & 
                         (df['assignment_group'] == assignment_group) & 
                         (df['priority'] == priority)]
        
        # Plot 1: Trending of Ticket Counts
        plt.figure(figsize=(10, 6))
        sns.countplot(data=filtered_df, x='month')
        plt.title(f'Ticket Counts by Month for {app_name}, {assignment_group}, Priority: {priority}')
        plt.xlabel('Month')
        plt.ylabel('Ticket Count')
        plt.xticks(rotation=45)
        plt.show()
        
        # Plot 2: Stacked Bar of Time-to-Resolve Bins
        grouped_data = filtered_df.groupby(['month', 'resolution_time_bucket']).size().unstack(fill_value=0)
        grouped_data.plot(kind='bar', stacked=True, figsize=(10, 6))
        plt.title(f'Resolution Time Buckets for {app_name}, {assignment_group}, Priority: {priority}')
        plt.xlabel('Month')
        plt.ylabel('Number of Tickets')
        plt.xticks(rotation=45)
        plt.legend(title='Resolution Time Bucket')
        plt.tight_layout()
        plt.show()

def update_plot(*args):
    plot_data(dropdown_app_name.value, dropdown_assignment_group.value, dropdown_priority.value)

# Observing changes
dropdown_app_name.observe(update_plot, 'value')
dropdown_assignment_group.observe(update_plot, 'value')

display(dropdown_app_name, dropdown_assignment_group, output)
update_plot()  # Initial call to display the plot

display(dropdown_app_name, dropdown_assignment_group, dropdown_priority, output)
update_plot()  # Initial call to display the plots


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

def update_plot(*args):
    plot_data(dropdown_app_name.value, dropdown_assignment_group.value, dropdown_priority.value)

# Linking the dropdown to the update function
dropdown_app_id.observe(update_plots, names='value')
dropdown_priority.observe(update_plot, 'value')

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
