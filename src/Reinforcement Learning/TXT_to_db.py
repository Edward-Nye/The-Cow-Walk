import sqlite3
import csv
from tqdm import tqdm

# Define the file paths
txt_file_path = 'TXT/MOVES.txt'
txt_file_path2 = 'TXT/cow_arrival_iterations.txt'  
db_file_path = 'MOVES.db'





# Connect to the SQLite database (it will be created if it doesn't exist)
conn = sqlite3.connect(db_file_path)
cursor = conn.cursor()
# Retrieve the list of all tables in the database
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
        
    # Loop through the tables and empty each one
for table in tables:
    table_name = table[0]
    cursor.execute(f"DELETE FROM {table_name};")
    print(f"Emptied table {table_name}")
        
        
conn.commit()
    


# Create a table in the SQLite database
cursor.execute('''
CREATE TABLE IF NOT EXISTS cow_data (
    cow_number INTEGER,
    state_0 INTEGER,
    state_1 INTEGER,
    state_2 INTEGER,
    state_3 INTEGER,
    state_4 INTEGER,
    state_5 INTEGER,
    action INTEGER,
    next_state_0 INTEGER,
    next_state_1 INTEGER,
    reward REAL,
    iteration INTEGER,
    frame INTEGER
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS cow_info (
    cow_number INTEGER,
    cow_name TEXT,
    walk_speed REAL,
    eagerness REAL,
    luck REAL,
    position INTEGER,
    exit_frame INTEGER,
    iteration INTEGER
)
''')

# Open the text file and read its contents
with open(txt_file_path, 'r') as file:
    lines = file.readlines()

# Iterate over the lines with a progress bar
for line in tqdm(lines, desc="Inserting data", unit="lines"):
    # Split the line by commas
    values = line.strip().split(',')

    # Convert the values to appropriate types
    cow_number = int(values[0])
    state_0 = int(values[1])
    state_1 = int(values[2])
    state_2 = int(values[3])
    state_3 = int(values[4])
    state_4 = int(values[5])
    state_5 = int(values[6])
    action = int(values[7])
    next_state_0 = int(values[8])
    next_state_1 = int(values[9])
    reward = float(values[10])
    iteration = int(values[11])
    frame = int(values[12])
    

    # Insert the data into the SQLite table
    cursor.execute('''
    INSERT INTO cow_data (
        cow_number, state_0, state_1, state_2, state_3, state_4, state_5,
        action, next_state_0, next_state_1, reward, iteration, frame
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (cow_number, state_0, state_1, state_2, state_3, state_4, state_5, action,
          next_state_0, next_state_1, reward, iteration, frame))
    
with open(txt_file_path2, 'r') as file:
    lines = file.readlines()

# Iterate over the lines with a progress bar
for line in tqdm(lines, desc="Inserting data", unit="lines"):
    # Split the line by commas
    values = line.strip().split(',')

    # Convert the values to appropriate types
    cow_number = int(values[0])
    cow_name = str(values[1])
    walk_speed = float(values[2])
    eagerness = float(values[3])
    luck = float(values[4])
    position = int(values[5])
    exit_frame = int(values[6])
    iteration = int(values[7])

    # Insert the data into the SQLite table
    cursor.execute('''
    INSERT INTO cow_info (
        cow_number, cow_name, walk_speed, eagerness, luck, position, exit_frame,
        iteration
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (cow_number, cow_name, walk_speed, eagerness, luck, position, exit_frame,
        iteration))

# Commit the transaction and close the connection
conn.commit()
conn.close()

print("Data successfully inserted into the database!")
