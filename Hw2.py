import turtle
import math
def draw_tree(t, l, depth):
    if depth == 0:
        return
    for  _ in range(4):
        t.forward(1)
        t.left(90)
    new_l = l * math.sqrt(2) / 2
    t.forward(new_l)
    draw_tree(t, new_l, depth - 1)
    t.backward(l)
    t.left(45)
    t.backward(l)
def main():
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
    'E': []
}
def bfs_with_animation(graph, start_node, ax, fig):
    visited = {node: False for node in graph}
    colors = {node: 'lightblue' for node in graph}
    stack = [start_node]
    visited[start_node] = True
    colors[start_node] = 'Skyblue'
    path_history = []
    while stack:
        current_node = stack.pop(0)
        path_history.append(current_node)
        colors[current_node] = 'lightgreen'
        for neighbor in graph[current_node]:
            if not visited[neighbor]:
                visited[neighbor] = True
                colors[neighbor] = 'Skyblue'
                stack.append(neighbor)
    return path_history
def animate_bfs(graph, start_node):
    visited = {node: False for node in graph}
    colors = {node: 'lightblue' for node in graph}
    path_history = []
    while queue:
        current_node = queue.popleft()
        visited[current_node] = True
        path_history.append(current_node)
        colors[current_node] = 'red'
        path_history.append(colors.copy())


        for neighbor in graph[current_node, []]:
            if not visited[neighbor]:
                visited[neighbor] = True
                colors[neighbor] = 'Skyblue'  
                queue.append(neighbor)
        colors[current_node] = 'black'
        path_history.append(colors.copy())
        return path_history
    def visualize_bfs(graph, start_node):
        G = nx.Graph(graph)
        pos = nx.bfs_layout(G, start_node)
        fig, ax = plt.subplots(figsize=(8, 6))
        plt.title(title)
        history = traversal_func(graph, start_node, ax, fig)
        def update(frame):
            ax.clear()
            current_colors = [history[frame].get(node, 'lightblue') for node in G.nodes()]

            nx.draw(G, pos, with_labels=True, node_color=current_colors, node_size=1000, font_size=10,font_weight='bold', arrowsize=10, ax=ax)
            ax.set_title(f"{title} (Step {frame+1}\(len(history)))", 
        repeatsize=16)
        ani = animation.FuncAnimation(fig, update, frames=len(history), repeat=False, interval=500)
        plt.show()
start_node = 'A'


        #####

    



    