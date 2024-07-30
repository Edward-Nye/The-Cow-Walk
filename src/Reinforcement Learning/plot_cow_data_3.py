import sqlite3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

def fetch_data(db_path, iterations):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    iteration_placeholders = ', '.join('?' for _ in iterations)
    query = f"""
    SELECT state_0, state_1, iteration, mode_action
    FROM V_Mode_Action_Per_Spot
    WHERE iteration IN ({iteration_placeholders})
    """
    
    cursor.execute(query, iterations)
    data = cursor.fetchall()
    
    conn.close()
    return data

def plot_colormesh(data, iterations):
    fig, axes = plt.subplots(1, len(iterations), figsize=(15, 5), constrained_layout=True)
    
    if len(iterations) == 1:
        axes = [axes]
    
    for ax, iteration in zip(axes, iterations):
        iteration_data = [row for row in data if row[2] == iteration]
        if not iteration_data:
            continue
            
        x = [row[0] for row in iteration_data]
        y = [row[1] for row in iteration_data]
        z = [row[3] for row in iteration_data]
        
        xi = np.array(sorted(list(set(x))))
        yi = np.array(sorted(list(set(y))))
        zi = np.zeros((len(yi), len(xi)))
        
        for row in iteration_data:
            x_idx = np.where(xi == row[0])[0][0]
            y_idx = np.where(yi == row[1])[0][0]
            zi[y_idx, x_idx] = row[3]
        
        pcm = ax.pcolormesh(xi, yi, zi, shading='auto', norm=Normalize(vmin=min(z), vmax=max(z)), cmap='viridis')
        ax.set_title(f'Iteration {iteration} #200')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.invert_yaxis()
        fig.colorbar(pcm, ax=ax, extend='both')
    
    plt.show()

# Example usage:
db_path = 'MOVES.db'
iterations_to_plot = [0, 5, 10, 15, 30, 49]  # Specify the iterations you want to plot

data = fetch_data(db_path, iterations_to_plot)
plot_colormesh(data, iterations_to_plot)
