
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import gym
from gym import spaces
from collections import defaultdict

class Cow:
    def __init__(self, number, name, walk_speed, eagerness, luck):
        self.number = number
        self.name = name
        self.walk_speed = walk_speed
        self.eagerness = eagerness
        self.luck = luck
        self.position = (0, 0)

        self.q_table = {}  # Initialize an empty Q-table

        self.learning_rate = 0.1
        self.discount_factor = 0.99
        self.exploration_rate = 0.70
    
        def get_state(self, herd):
       # Ensure state representation includes position as tuple elements
        nearby_cows = [(other_cow.position[0] - self.position[0], other_cow.position[1] - self.position[1]) for other_cow in herd if other_cow != self]
        state = (self.position[0], self.position[1], self.walk_speed, self.eagerness) + tuple(np.array(nearby_cows).flatten())
        return state

    def choose_action(self, state):
        if state not in self.q_table:
        # Initialize the Q-values for the new state
            self.q_table[state] = [0, 0, 0, 0]
        # Set action 1 to have a high initial Q-value
            self.q_table[state][1] = 0.1
        if random.random() < self.exploration_rate:  # Exploration
            return random.randint(0, 3)
        return np.argmax(self.q_table[state])
    


    def learn(self, state, action, reward, next_state, done):
        old_q_value = self.q_table.get(state, [0, 0, 0, 0])[action]
        next_max = np.max(self.q_table.get(next_state, [0, 0, 0, 0]))
        
        new_q_value = old_q_value + self.learning_rate * (reward + self.discount_factor * next_max * (1 - done) - old_q_value)
        
        if state not in self.q_table:
            self.q_table[state] = [0, 0, 0, 0]
        
        self.q_table[state][action] = new_q_value

    def calculate_starting_position(self, paddock_width, paddock_height):
        eagerness_norm = self.eagerness / 100
        luck_norm = self.luck / 100

        x = random.randint(0, paddock_width - 1)
        y = paddock_height/(np.exp(((eagerness_norm + luck_norm)**2.5)/2))

        return int(x), max(0, min(int(y), paddock_height - 1))
    
    def move_towards(self, target, paddock_width, paddock_height):
        x, y = self.position
        target_x, target_y = target

        move_step = 2 ** (self.walk_speed / 100)

        # Determine movement in x direction    
        if x < target_x:
            x += move_step
        elif x > target_x:
            x -= move_step

        # Determine movement in y direction
        if y < target_y:
            y += move_step
        elif y > target_y:
            y -= move_step 
        

            
        # Ensure the new position is within paddock bounds
        x = max(0, min(x, paddock_width - 1))
        y = max(0, min(y, paddock_height - 1))

        self.position = (int(x), int(y))  # Update position


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

        #x, y = cow.position
        #self.grid[y, x] = 0  # Mark the cell as unoccupied by a cow
    

    #def new_positions(self, herd, frame_number, iteration):
    #    grid_str = "\n".join(" ".join(map(str, row)) for row in self.grid)
    #    with open(f"TXT/paddock_frames_{iteration}.txt", "a") as file:
    #        file.write(f"TXT/Frame {frame_number}:\n{grid_str}\n\n")

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
    
    def update_positions(self, herd, target, arrived, frame_number, iteration):
        new_positions = set()
        
        target_x, target_y = target
        herd.sort(key=lambda cow: abs(cow.position[0] - target_x) + abs(cow.position[1] - target_y))
        
          # Track new positions for each cow
        for cow in herd[:]:  # Use a slice to iterate over a copy of the herd
            self.remove_cow(cow)  # Remove current position
            old_position = cow.position  # Store old position
            
            # Try to move cow towards the target
            cow.move_towards(target, self.width, self.height)
            new_position = cow.position
            
            # Check if the new position is already occupied or if it's in new_positions
            if (self.grid[new_position[1], new_position[0]] == 0) and (new_position not in new_positions):
                new_positions.add(new_position)  # Mark this new position as occupied
                if new_position[0] <= 5 and new_position[1] == 0:
                    arrived.append(cow)  # Add to the arrived list
                    herd.remove(cow)  # Remove from the herd
                else:
                    self.place_cow(cow)  # Update new position
            else:
                cow.position = old_position  # Revert to old position if new position is occupied
                self.place_cow(cow)  # Place the cow back at the old position
        
        # Save the current frame
        self.save_frame(frame_number, iteration)

    def save_frame(self, frame_number, iteration):
        grid_str = "\n".join(" ".join(map(str, row)) for row in self.grid)
        with open(f"TXT/paddock_frames_{iteration}.txt", "a") as file:
            file.write(f"TXT/Frame {frame_number}:\n{grid_str}\n\n")
        #print(f"Saved frame {frame_number}")

    def display(self):
        print(self.grid)


class PaddockEnv(gym.Env):
    def __init__(self, width, height, target):
        super(PaddockEnv, self).__init__()
        self.width = width
        self.height = height
        self.target = target

        # Define action space: [up, down, left, right]
        self.action_space = spaces.Discrete(4)

        # Define observation space: [cow_x, cow_y]
        self.observation_space = spaces.Box(low=0, high=max(width, height), shape=(2,), dtype=np.int32)
        
        # Initial state
        self.state = None
        self.reset()

    def reset(self):
        self.state = np.array([np.random.randint(0, self.width), np.random.randint(0, self.height)])
        return self.state

    def step(self, action):
        x, y = self.state
        
        if action == 0:  # up
            y = min(y + 1, self.height - 1)
        elif action == 1:  # down
            y = max(y - 1, 0)
        elif action == 2:  # left
            x = max(x - 1, 0)
        elif action == 3:  # right
            x = min(x + 1, self.width - 1)
        
        self.state = np.array([x, y])
        
        # Calculate reward
        if (x, y) == self.target:
            reward = 10
            done = True
        else:
            reward = -1
            done = False
        
        return self.state, reward, done, {}

    def render(self, mode='human'):
        paddock = np.zeros((self.height, self.width), dtype=int)
        paddock[self.state[1], self.state[0]] = 1
        print(paddock)
    
class QLearningAgent:
    def __init__(self, action_space, learning_rate=0.1, discount_factor=0.99, exploration_rate=1.0, exploration_decay=0.995):
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.q_table = defaultdict(lambda: np.zeros(action_space.n))

    def choose_action(self, state):
        if random.uniform(0, 1) < self.exploration_rate:
            return self.action_space.sample()
        return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state, done):
        q_predict = self.q_table[state][action]
        if done:
            q_target = reward
        else:
            q_target = reward + self.discount_factor * np.max(self.q_table[next_state])
        self.q_table[state][action] += self.learning_rate * (q_target - q_predict)

        if done:
            self.exploration_rate *= self.exploration_decay

def save_cows_to_file(herd, filename):
    with open(filename, "w") as file:
        for cow in herd:
            file.write(f"{cow.number},{cow.name},{cow.walk_speed},{cow.eagerness},{cow.luck}\n")

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
            #if i < midpoint:
            #    position_factor = 1 - (i / midpoint)
            #else:
            #    position_factor = -((i - midpoint) / midpoint)
            
            
            
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
            cow.walk_speed -= 0.1
        
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

    if action == 0:  # up
        next_y = min(y + 1, paddock.height - 1)
    elif action == 1:  # down
        next_y = max(y - 1, 0)
    elif action == 2:  # left
        next_x = max(x - 1, 0)
    elif action == 3:  # right
        next_x = min(x + 1, paddock.width - 1)

    # Check if the new position is occupied by another cow
    if any((other_cow.position[0], other_cow.position[1]) == (next_x, next_y) for other_cow in herd):
        reward = -5  # Penalty for trying to move into an occupied space
        next_x, next_y = x, y  # Stay in the current position
    else:
        reward = -1  # Small penalty for each move

    next_state = (next_x, next_y, cow.walk_speed, cow.eagerness)
    done = False
    
    if next_x == any(paddock.target[0]) and next_y == paddock.target[1]:
        reward = 10  # Reward for reaching the target
        done = True

    return next_state, reward, done

def main():
    paddock_width = 25
    paddock_height = 50
    target = (np.arange(5), 0)
    iterations = 20
    
    herd = load_cows_from_file("TXT/cows.txt")

  
    for iteration in range(iterations):
        paddock = Paddock(paddock_width, paddock_height)
        
        arrived = []
        
        cow_drop(paddock, herd)

        paddock.target = target
        frame_number = 0
        
        open(f"TXT/paddock_frames_{iteration}.txt", "w").close()
        
        with tqdm(total=len(herd), desc=f"Moving cows iteration {iteration + 1}") as pbar:
            while herd:
                
                for cow in herd:   
                    state = cow.get_state(herd)
                    action = cow.choose_action(state)
                    #print(f"Cow {cow.number} at {cow.position} takes action {action}")  # Debug action selection
                    next_state, reward, done = env_step(paddock, cow, action, herd)
                    #print(f"Cow {cow.number} moves to {next_state[:2]} with reward {reward}")  # Debug position update
                    cow.learn(state, action, reward, next_state, done)
                    cow.position = (next_state[0], next_state[1])


                    if done:
                        arrived.append(cow)
                        herd.remove(cow)
                        
                frame_number += 1
                pbar.update(len(arrived) - pbar.n)
                paddock.new_positions(herd, frame_number, iteration)
                #print(f"Arrived cows: {[cow.name for cow in arrived]}")

               
        if iteration == 0:
            herd = load_cows_from_file("TXT/cows.txt")

        else:
            herd = load_cows_from_file(f"TXT/cows_after_iteration_{iteration-1}.txt")

        average_walk_speed = sum(cow.walk_speed for cow in herd) / len(herd)
        print(average_walk_speed)
        update_walk_speed(herd, arrived, average_walk_speed)
        update_eagerness(herd)

        save_cows_to_file(arrived, f"TXT/cow_arrival_iteration_{iteration}.txt")
        
        save_cows_to_file(herd, f"TXT/cows_after_iteration_{iteration}.txt")

if __name__ == "__main__":
    main()
