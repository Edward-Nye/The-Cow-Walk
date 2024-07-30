
import random
import numpy as np
import json
from tqdm import tqdm


class Cow:
    def __init__(self, number, name, walk_speed, eagerness, luck):
        self.number = number
        self.name = name
        self.walk_speed = walk_speed
        self.eagerness = eagerness
        self.luck = luck
        self.position = (0, 0)
        self.exitframe = None

        self.q_table = {}  # Initialize an empty Q-table

        self.learning_rate = 0.9
        self.discount_factor = 0.1
        self.exploration_rate = None
    
    def initialize_q_table(self, paddock_width, paddock_height):
        for x in range(paddock_width):
            for y in range(paddock_height):
                state = (x, y, 0, 0, 0, 0)  # Example state structure
                self.q_table[state] = [random.uniform(-1, 1) for _ in range(4)]
    
    def get_state(self, herd, paddock):
    # Define the positions of the spaces around the cow
        left = (self.position[0] - 1, self.position[1])
        right = (self.position[0] + 1, self.position[1])
        up = (self.position[0], self.position[1] + 1)
        down = (self.position[0], self.position[1] - 1)
    
        # Helper function to determine if a space is occupied
        def is_occupied(position):
            x, y = position
            if x < 0 or x >= paddock.width or y < 0 or y >= paddock.height:
                return 1  # Consider out-of-bounds as occupied
            for other_cow in herd:
              if other_cow.position == position:
                   return 1  # Occupied
            return 0  # Unoccupied

        # Determine the condition of the spaces around the cow
        left_condition = is_occupied(left)
        right_condition = is_occupied(right)
        up_condition = is_occupied(up)
        down_condition = is_occupied(down)

        # Create the state representation
        state = [self.position[0], self.position[1], left_condition, right_condition, up_condition, down_condition]
        return state

    def choose_action(self, state, iteration):
        state = tuple(state)

        if iteration < 9:
            self.exploration_rate = 1 - iteration/10
        else:
            self.exploration_rate = 0.01
       
        if state not in self.q_table:
        # Set inital q state
            self.q_table[state] = np.random.uniform(low=0, high=1, size=4).tolist()
        if random.random() < self.exploration_rate:  # Exploration
            return random.randint(0, 3)
        else:
            return np.argmax(self.q_table.get(state, [0, 0, 0, 0]))
    


    def learn(self, state, action, reward, next_state, done):
        state = tuple(state)  # Ensure state is a tuple
        next_state = tuple(next_state)  # Ensure next_state is a tuple
        
        old_q_value = self.q_table.get(state, [0, 0, 0, 0])[action]
        next_max = np.max(self.q_table.get(next_state, [0, 0, 0, 0]))
        
        new_q_value = old_q_value + self.learning_rate * (reward + self.discount_factor * next_max * (1 - done) - old_q_value)
        
        if state not in self.q_table:
            self.q_table[state] = np.random.uniform(low=-1, high=1, size=4).tolist()
        
        self.q_table[state][action] = new_q_value  

    def calculate_starting_position(self, paddock_width, paddock_height):
        eagerness_norm = self.eagerness / 100
        luck_norm = self.luck / 100

        x = random.randint(0, paddock_width - 1)
        y = paddock_height/(np.exp(((eagerness_norm + luck_norm)**2.5)/2))

        return int(x), max(0, min(int(y), paddock_height - 1))
    


class Paddock:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width), dtype=int)  # Initialize paddock grid
        self.target = None

    def place_cow(self, cow):
        x, y = cow.position
        self.grid[y, x] = cow.number  # Mark the cell with the cow's number

    def remove_cow(self, cow):
        for x in range(self.width):
            for y in range(self.height):
                if self.grid[y, x] == cow.number:
                    self.grid[y, x] = 0

    def new_positions(self, herd, frame_number, iteration):
        for cow in herd:
            self.place_cow(cow)

        grid_str = "\n".join(" ".join(map(str, row)) for row in self.grid)
        with open(f"TXT/paddock_frames_{iteration}.txt", "a") as file:
            file.write(f"Frame {frame_number}:\n{grid_str}\n\n")
            
        for cow in herd:
            self.remove_cow(cow)

    def close_sort(self, herd, target):
        target_x, target_y = target
        herd.sort(key=lambda cow: abs(cow.position[0] - target_x) + abs(cow.position[1] - target_y))
        return herd
    
    def save_frame(self, frame_number, iteration):
        grid_str = "\n".join(" ".join(map(str, row)) for row in self.grid)
        with open(f"TXT/paddock_frames_{iteration}.txt", "a") as file:
            file.write(f"TXT/Frame {frame_number}:\n{grid_str}\n\n")

    def display(self):
        print(self.grid)


def save_q_tables(cow, frame, iteration, filename):
    with open(filename, "a") as file:
        for state, q_table in cow.q_table.items():
            state_json = json.dumps(state)
            q_table_json = json.dumps(q_table)

            file.write(f"{cow.number},{state_json},{frame},{iteration},,{q_table_json}\n")
        
    

def save_cows_to_file(herd, iteration, mode, filename):
    if mode == 0:
        with open(filename, "w") as file:
            for cow in herd:
                file.write(f"{cow.number},{cow.name},{cow.walk_speed},{cow.eagerness},{cow.luck}\n")
    if mode == 1:
        with open(filename, "a") as file:
            for cow in herd:
                position = herd.index(cow)
                file.write(f"{cow.number},{cow.name},{cow.walk_speed},{cow.eagerness},{cow.luck},{position+1},{cow.exitframe},{iteration+1}\n")

def moveSave(cow, state, action, nextState, reward, iteration, frame, filename):
    with open(filename, "a") as file:
        file.write(f"{cow.number},{state[0]},{state[1]},{state[2]},{state[3]},{state[4]},{state[5]},{action},{nextState[0]},{nextState[1]},{reward},{iteration},{frame}\n")

def load_cows_from_file(filename):
    cows = []
    with open(filename, "r") as file:
        for line in file:
            number, name, walk_speed, eagerness, luck = line.strip().split(",")
            cow = Cow(int(number), name, float(walk_speed), float(eagerness), float(luck))
            cows.append(cow)
    return cows

def update_walk_speed(herd, arrived, average_walk_speed):
    
    arrival_order = {cow.number: i for i, cow in enumerate(arrived)}
    midpoint = len(herd) // 2

    for cow in herd:
        
        if cow.number in arrival_order:
            i = arrival_order[cow.number]
            
            if i < midpoint and cow.walk_speed < average_walk_speed:
                position_factor = 1 - (i / midpoint)
                multiplier = ((200 - cow.walk_speed) / 100)
            
            if i < midpoint and cow.walk_speed > average_walk_speed:
                position_factor = 1 - (i / midpoint)
                multiplier = ((100 - cow.walk_speed) / 100)
            
            if i > midpoint and cow.walk_speed < average_walk_speed:
                position_factor = -((i - midpoint) / midpoint)
                multiplier = ((100 - cow.walk_speed) / 100)
            
            if i > midpoint and cow.walk_speed > average_walk_speed:
                position_factor = -((i - midpoint) / midpoint)
                multiplier = ((200 - cow.walk_speed) / 100)
            else:
                position_factor = 0
                multiplier = 0
            
            
        else:
            cow.walk_speed -= 0.1
            position_factor = 0
            multiplier = 0
        
        change = position_factor * multiplier
        cow.walk_speed += change

        cow.walk_speed = min(max(cow.walk_speed, 0), 100)

def update_eagerness(herd):
    for cow in herd:
        change = random.uniform(-5, 5)
        cow.eagerness += change
        cow.eagerness = min(max(cow.eagerness, 0), 100)

def cow_drop(paddock, herd):
    occupied_positions = set()
    for cow in herd:
        x, y = cow.calculate_starting_position(paddock.width, paddock.height)
        
        if (x, y) in occupied_positions:
            found_position = False
            for dx in range(-1,1):
                for dy in range(-1,1):
                    new_x, new_y = x + dx, y + dy
                    if 0 <= new_x < paddock.width and 0 <= new_y < paddock.height:
                        if (new_x, new_y) not in occupied_positions:
                            x, y = new_x, new_y
                            found_position = True
                            break
                if found_position:
                    break
        
        if (x, y) in occupied_positions:
            while (x, y) in occupied_positions:
                x = random.randint(0, paddock.width - 1)
                y = random.randint(0, paddock.height -1)
       
        cow.position = (x, y)
        paddock.place_cow(cow)
        occupied_positions.add((x, y))

    return paddock

def env_step(paddock, cow, action, herd):
    x, y = cow.position
    next_x, next_y = x, y

    reward = 0

    if action == 0:  # Down
        next_y = min(y + 1, paddock.height - 1)
    elif action == 1:  #UP
        next_y = max(y - 1, 0)
    elif action == 2:  # left
        next_x = max(x - 1, 0)
    elif action == 3:  # right
        next_x = min(x + 1, paddock.width - 1)

    # Check if the new position is occupied by another cow
    if any((other_cow.position[0], other_cow.position[1]) == (next_x, next_y) for other_cow in herd):
        reward -=10  # Penalty for trying to move into an occupied space
        next_x, next_y = x, y  # Stay in the current position
    else:
        reward -= 1  # Small penalty for each move

    # Proximity reward/penalty
    target_x, target_y = paddock.target
    dist_before_x = abs(x - target_x)
    dist_after_x = abs(next_x - target_x)
    dist_before_y = abs(y - target_y)
    dist_after_y = abs(next_y - target_y)

    if any(dist_after_x < dist_before_x) and dist_after_y < dist_before_y:
        reward += 5  # Reward for moving closer to the target X & Y
    elif any(dist_after_x < dist_before_x):
        reward += 1  # Reward for moving closer to the target X
    elif dist_after_y < dist_before_y:
        reward += 2  # Reward for moving closer to the target Y
    else:
        reward -= 1  # Penalty for moving away from the target

    # Exploration penalty
    if (next_x, next_y) == (x, y):
        reward -= 2  # Penalty for not moving

    next_state = (next_x, next_y, cow.walk_speed, cow.eagerness)
    done = False

    # Check if the cow reaches the target
    if next_x == any(target_x) and next_y == target_y:
        reward += 10  # Reward for reaching the target
        done = True


    return next_state, reward, done


def main(iterations=1):

    NoCow = 200
    paddock_width = 25
    paddock_height = 75
    target = (np.arange(5), 0)
    
    herd = load_cows_from_file("TXT/cows.txt")
    herd = herd[:NoCow]
    
    open(f"TXT/MOVES.txt", "w").close()
    #open(f"TXT/Qtables.txt", "w").close()
    open(f"TXT/cow_arrival_iterations.txt", "w").close()
    
    #for cow in herd:
    #    cow.initialize_q_table(paddock_width, paddock_height)

    for iteration in range(iterations):
        paddock = Paddock(paddock_width, paddock_height)
        herd = herd[:NoCow]

        arrived = []
        
        cow_drop(paddock, herd)

        paddock.target = target
        frame_number = 0
        
        open(f"TXT/paddock_frames_{iteration}.txt", "w").close()
        
        with tqdm(total=len(herd), desc=f"Moving cows iteration {iteration + 1}") as pbar:
            while herd:
                
                for cow in herd:   
                    state = cow.get_state(herd, paddock)
                    action = cow.choose_action(state, iteration)
                    #print(f"Cow {cow.number} at {cow.position} takes action {action}")  # Debug action selection
                    next_state, reward, done = env_step(paddock, cow, action, herd)
                    #print(f"Cow {cow.number} moves to {next_state[:2]} with reward {reward}")  # Debug position update
                    cow.learn(state, action, reward, next_state, done)
                    cow.position = (next_state[0], next_state[1])
                    moveSave(cow, state, action, next_state[:2], reward, iteration, frame_number, f"TXT/MOVES.txt")
                    #save_q_tables(cow, frame_number, iteration, f"TXT/Qtables.txt")

                    if done:
                        arrived.append(cow)
                        cow.exitframe = frame_number
                        herd.remove(cow)
                        
                frame_number += 1
                pbar.update(len(arrived) - pbar.n)
                paddock.new_positions(herd, frame_number, iteration)
                #print(f"Arrived cows: {[cow.name for cow in arrived]}")

               
        if iteration == 0:
            herd = load_cows_from_file("TXT/cows.txt")
            herd = herd[:NoCow]

        else:
            herd = load_cows_from_file(f"TXT/cows_after_iteration_{iteration-1}.txt")
            herd = herd[:NoCow]

        average_walk_speed = sum(cow.walk_speed for cow in herd) / len(herd)
        print(average_walk_speed)
        update_walk_speed(herd, arrived, average_walk_speed)
        update_eagerness(herd)

        save_cows_to_file(arrived, iteration, 1, f"TXT/cow_arrival_iterations.txt")
        
        save_cows_to_file(herd, iteration, 0, f"TXT/cows_after_iteration_{iteration}.txt")
    
if __name__ == "__main__":
    iterations = int(input("How many iterations: "))
    main(iterations)
