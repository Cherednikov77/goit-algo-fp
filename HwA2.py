import turtle
import math
def draw_tree(t, l, depth):
    if depth == 0:
        return
    for  _ in range(4):
        t.forward(l)
        t.left(90)
    new_l = l * math.sqrt(2) / 2
    t.forward(l)
    t.left(45)
    draw_tree(t, new_l, depth - 1)
    t.right(90)
    t.forward(new_l)
    draw_tree(t, new_l, depth - 1)
    t.backward(new_l)
    t.left(45)
    t.backward(l)
    
def main():
    try:
        user_depth = int(input("Enter the depth of the tree (non-negative integer): "))
    except ValueError:
        print("Invalid input. Please enter a non-negative integer.")
        user_depth = 7
    
    screen = turtle.Screen()
    screen.tracer(0)
    t = turtle.Turtle()
    t.speed(0)
    t.left(90)

    draw_tree(t, 100, 10)
    screen.update()
    screen.mainloop()

if __name__ == "__main__":
    main()


    #####
import heapq
def dijkstra(graph, start):
    queue = []
    heapq.heappush(queue, (0, start))
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    while queue:
        current_distance, current_node = heapq.heappop(queue)
        if current_distance > distances[current_node]:
            continue
        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(queue, (distance, neighbor))
    return distances
graph = {
    'A': {'B': 1, 'C': 4},
    'B': {'A': 1, 'C': 2, 'D': 5},
    'C': {'A': 4, 'B': 2, 'D': 1},
    'D': {'B': 5, 'C': 1}
}

print(dijkstra(graph, "A"))

#####
import networkx as nx
import matplotlib.pyplot as plt


G = nx.Graph()
for node, edges in graph.items():
    for neighbor, weight in edges.items():
        G.add_edge(node, neighbor, weight=weight)

pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True)
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
plt.show()

######
import random
import matplotlib.pyplot as plt
def monte_carlo_dice(n_simulations=1000):
    results = {i: 0 for i in range(2, 13)}
    for _ in range(n_simulations):
        die1 = random.randint(1, 6)
        die2 = random.randint(1, 6)
        total = die1 + die2
        results[total] += 1
    probabilities = {s: count / n_simulations for s, count in results.items()}
    return results, probabilities
n = 1000 
counts, probs = monte_carlo_dice(n)
print(f"{'sum':<10} {'frequency':<10} {'probability (%)':<15}")
print(f"{s:<10} {counts[s]:<10} {probs[s]*100:<15.2f}" for s in range(2, 13))
plt.figure(figsize=(10, 6))
plt.bar(counts.keys(), counts.values(), width=0.6, color='skyblue', edgecolor='black')
plt.xlabel('Sum of Two Dice')
plt.ylabel('Frequency')
plt.title(f'Dice Roll Simulation Results (n={n})')
plt.xticks(range(2, 13))
plt.show()

######

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import time
tree_graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F', 'G'],
    'D': [],
    'E': [],
    'F': [],
    'G': []
}
def generate_colors(steps):
    colors = []
    for step in range(steps):
        step_colors = {}
        for node in tree_graph:
            if node in step['visited']:
                step_colors[node] = 'lightgreen'
            elif node in step['queue']:
                step_colors[node] = 'Skyblue'
            else:
                step_colors[node] = 'lightblue'
        colors.append(step_colors)
    return colors


def bfs_path_history(graph, start_node):
    visited = []
    queue = deque([start_node])
    history = []
    current_colors = {node: 'lightblue' for node in graph}
    while queue:
        current_node = queue.popleft()
        if current_node not in visited:
            visited.append(current_node)
            current_colors[current_node] = 'lightgreen'
    for neighbor in graph[current_node]:
        if neighbor not in visited:
                queue.append(neighbor)
                current_colors[neighbor] = 'Skyblue'
    history.append(current_colors.copy())
    current_colors[current_node] = 'black'
    return history
def visualize_traversal(graph, start_node, traversal_func, title="BFS Animation"):
    G = nx.Graph(graph)
    pos = nx.spring_layout(G)
    fig, ax = plt.subplots(figsize=(8, 6))
    history = bfs_path_history(graph, start_node)
    def update(frame):
        ax.clear()
        node_colors = [history[frame].get(node, 'lightblue') for node in G.nodes()]
        nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=1000, font_size=10, font_weight='bold', arrowsize=10, ax=ax)
        ax.set_title(f"{title} (Step {frame + 1}/{len(history)}", fontsize=16)
    ani = animation.FuncAnimation(fig, update, frames=len(history), repeat=False, interval=500)
    plt.show()
if __name__ == "__main__":
    start_node = 'A'             
    visualize_traversal(tree_graph, start_node, "BFS")
    time.sleep(1)
    visualize_traversal(tree_graph, start_node, "DFS")

   
#####
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque


tree_graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F', 'G'],
    'D': [], 'E': [], 'F': [], 'G': []
}

def generate_colors(steps):
    colors = []
    for i in range(steps):
        level = int(150 + (105 * (i / steps))) if steps > 1 else 200
        
        hex_color = f"#{i*20:02x}{level//2:02x}{level:02x}" 
        
        r = int(18 + (180 * (i / steps)))
        g = int(150 + (80 * (i / steps)))
        b = int(240)
        colors.append(f"#{r:02x}{g:02x}{b:02x}")
    return colors

def get_bfs_history(graph, start_node):
    
    visited = []
    queue = deque([start_node])
    history = []
    
    
    order = []
    temp_queue = deque([start_node])
    temp_visited = set()
    while temp_queue:
        node = temp_queue.popleft()
        if node not in temp_visited:
            temp_visited.add(node)
            order.append(node)
            temp_queue.extend(graph.get(node, []))

    colors_list = generate_colors(len(order))
    node_colors = {node: "#E6F2FF" for node in graph} 

    for i, node in enumerate(order):
        node_colors[node] = colors_list[i]
        history.append(node_colors.copy())
    
    return history

def get_dfs_history(graph, start_node):
    
    visited = []
    stack = [start_node]
    history = []
    
    order = []
    temp_stack = [start_node]
    temp_visited = set()
    while temp_stack:
        node = temp_stack.pop()
        if node not in temp_visited:
            temp_visited.add(node)
            order.append(node)
            
            neighbors = list(graph.get(node, []))
            temp_stack.extend(reversed(neighbors))

    colors_list = generate_colors(len(order))
    node_colors = {node: "#E6F2FF" for node in graph}

    for i, node in enumerate(order):
        node_colors[node] = colors_list[i]
        history.append(node_colors.copy())
        
    return history

def visualize_traversal(graph_data, start_node, mode="BFS"):
    G = nx.Graph(graph_data)
    
   
    pos = {
        'A': (0, 0),
        'B': (-2, -1), 'C': (2, -1),
        'D': (-3, -2), 'E': (-1, -2), 'F': (1, -2), 'G': (3, -2)
    }

    if mode == "BFS":
        history = get_bfs_history(graph_data, start_node)
    else:
        history = get_dfs_history(graph_data, start_node)

    fig, ax = plt.subplots(figsize=(10, 7))
    
    def update(frame):
        ax.clear()
        current_colors = [history[frame].get(node, "#E6F2FF") for node in G.nodes()]
        nx.draw(G, pos, with_labels=True, node_color=current_colors, 
                node_size=2000, font_size=12, font_weight='bold', 
                edge_color='gray', width=2, ax=ax)
        ax.set_title(f"Обхід дерева: {mode} (Крок {frame + 1})", fontsize=16)

    ani = animation.FuncAnimation(fig, update, frames=len(history), repeat=False, interval=800)
    plt.show()

if __name__ == "__main__":
    visualize_traversal(tree_graph, 'A', "BFS")
    visualize_traversal(tree_graph, 'A', "DFS")

   