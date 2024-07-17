import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

# Connect to the database
db_path = 'MOVES.db'  # Update this path
conn = sqlite3.connect(db_path)

# Function to execute a SQL query and return a DataFrame
def query_to_df(query):
    return pd.read_sql_query(query, conn)

# Query to create the required view
create_view_query = """
CREATE VIEW IF NOT EXISTS action_reward_distribution AS
SELECT  
    action, reward, COUNT(*) AS Action_Reward
FROM 
    cow_data
GROUP BY
    reward, action;
"""

# Execute the view creation query
conn.execute(create_view_query)

# Query to retrieve data from the view
data_query = "SELECT * FROM action_reward_distribution;"

# Retrieve data into DataFrame
df = query_to_df(data_query)

# Close the database connection
conn.close()

# Plotting the pie charts
actions = df['action'].unique()
num_actions = len(actions)

plt.figure(figsize=(16, 8))

for i, action in enumerate(actions, 1):
    action_data = df[df['action'] == action]
    # Calculate the explode values (all slices will be exploded equally)
    explode = [0.1] * len(action_data)  # 0.1 is the fraction of the radius with which to offset each wedge

    plt.subplot(1, num_actions, i)
    plt.pie(action_data['Action_Reward'], labels=action_data['reward'], autopct='%1.1f%%', startangle=140, explode=explode)
    plt.title(f'Action {action} Reward Distribution #200')
    plt.legend()
plt.tight_layout()
plt.show()
