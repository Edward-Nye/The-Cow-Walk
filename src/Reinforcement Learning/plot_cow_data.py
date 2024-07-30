import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

# Connect to the database
db_path = 'MOVES.db'  # Update this path
conn = sqlite3.connect(db_path)

# Function to execute a SQL query and return a DataFrame
def query_to_df(query):
    return pd.read_sql_query(query, conn)

# Create the views if not already created
views_creation_queries = [
    """
    CREATE VIEW IF NOT EXISTS avg_speed_by_iteration AS
    SELECT iteration, AVG(walk_speed) AS avg_speed
    FROM cow_info
    GROUP BY iteration;
    """,
    """
    CREATE VIEW IF NOT EXISTS rewards_by_cow_iteration AS
    SELECT cd.cow_number, ci.cow_name, cd.iteration, SUM(cd.reward) AS total_reward
    FROM cow_data cd
    JOIN cow_info ci ON cd.cow_number = ci.cow_number AND cd.iteration = ci.iteration
    GROUP BY cd.cow_number, ci.cow_name, cd.iteration;
    """,
    """
    CREATE VIEW IF NOT EXISTS final_positions AS
    SELECT cow_number, cow_name, MAX(position) AS final_position
    FROM cow_info
    GROUP BY cow_number, cow_name;
    """,
    """
    CREATE VIEW IF NOT EXISTS action_distribution AS
    SELECT iteration, action, COUNT(*) AS action_count
    FROM cow_data
    GROUP BY iteration, action;
    """
]

# Execute the view creation queries
for query in views_creation_queries:
    conn.execute(query)

# Queries to retrieve data from views
queries = {
    'avg_speed_by_iteration': "SELECT * FROM avg_speed_by_iteration;",
    'rewards_by_cow_iteration': "SELECT * FROM rewards_by_cow_iteration;",
    'final_positions': "SELECT * FROM final_positions;",
    'action_distribution': "SELECT * FROM action_distribution;"
}

# Retrieve data into DataFrames
df_avg_speed = query_to_df(queries['avg_speed_by_iteration'])
df_rewards = query_to_df(queries['rewards_by_cow_iteration'])
df_final_positions = query_to_df(queries['final_positions'])
df_action_distribution = query_to_df(queries['action_distribution'])

# Close the database connection
conn.close()

# Plotting the graphs
plt.figure(figsize=(14, 10))

# Average Speed by Iteration
#plt.subplot(2, 2, 1)
#plt.plot(df_avg_speed['iteration'], df_avg_speed['avg_speed'], marker='o')
#plt.title('Average Speed by Iteration')
#plt.xlabel('Iteration')
#plt.ylabel('Average Speed')
#plt.grid(True)

# Total Rewards by Cow and Iteration
plt.subplot(2, 1, 2)
for cow_name in df_rewards['cow_name'].unique():
    cow_data = df_rewards[df_rewards['cow_name'] == cow_name]
    plt.bar(cow_data['iteration'], cow_data['total_reward'], label=cow_name)
plt.title('Total Rewards by Cow and Iteration #200')
plt.xlabel('Iteration')
plt.ylabel('Total Reward')
#plt.legend()
plt.grid(True)

# Final Positions of Cows
#plt.subplot(2, 2, 3)
#plt.bar(df_final_positions['cow_name'], df_final_positions['final_position'])
#plt.title('Final Positions of Cows')
#plt.xlabel('Cow Name')
#plt.ylabel('Final Position')
#plt.xticks(rotation=45)
#plt.grid(True)

# Actions Distribution by Iteration
plt.subplot(2, 1, 1)
for action in df_action_distribution['action'].unique():
    action_data = df_action_distribution[df_action_distribution['action'] == action]
    plt.bar(action_data['iteration'], action_data['action_count'], label=f'Action {action}')
plt.title('Actions Distribution by Iteration #200')
plt.xlabel('Iteration')
plt.ylabel('Action Count')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
