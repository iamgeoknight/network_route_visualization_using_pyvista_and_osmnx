import osmnx as ox
import pyvista as pv
import numpy as np
import random
import heapq
from functools import lru_cache


# Create a networkx graph from a place
place = "Chandigarh, India"
G = ox.graph_from_place(place, network_type="drive")

nodes, edges = ox.graph_to_gdfs(G)
nodes['osmid'] = nodes.index


def vistaLines(nodes, edges):
    # nodes, edges = ox.graph_to_gdfs(G)
    pts_list = edges['geometry'].apply(lambda g: np.column_stack(
        (g.xy[0], g.xy[1], np.zeros(len(g.xy[0]))))).tolist()
    vertices = np.concatenate(pts_list)

    lines = []  # Create an empty array with 3 columns

    j = 0

    for i in range(len(pts_list)):
        pts = pts_list[i]
        vertex_length = len(pts)
        vertext_start = j
        vertex_end = j + vertex_length - 1
        vertex_arr = [vertex_length] + \
            list(range(vertext_start, vertex_end + 1))
        lines.append(vertex_arr)
        j += vertex_length
    return pv.PolyData(vertices, lines=np.hstack(lines))


def get_start_goal_nodes():
    global G
    nodes = list(G.nodes)

    # Use two random nodes as start and goal (you can replace them with specific nodes if you want)
    start = random.choice(nodes)
    goal = random.choice(nodes)

    # Ensure start and goal nodes are not the same
    while start == goal:
        goal = random.choice(nodes)

    return start, goal


@lru_cache(maxsize=None)  # Add caching to the successors function
def successors(node):
    global G

    # Find the neighbors of the current node
    neighbors = list(G.neighbors(node))

    # Calculate the edge weight and add it as a tuple (neighbor, weight)
    successors = [(neighbor, G[node][neighbor][0]['length'])
                  for neighbor in neighbors]

    return successors


@lru_cache(maxsize=None)  # Add caching to the great_circle_distance function
def great_circle_distance(p1, p2):
    global G
    coord1 = G.nodes[p1]['y'], G.nodes[p1]['x']
    coord2 = G.nodes[p2]['y'], G.nodes[p2]['x']
    return ox.distance.great_circle(*coord1, *coord2)


def dijkstra(start, goal):
    # Initialize variables
    # Distances from start to each node
    distances = {node: float('inf') for node in G.nodes}
    distances[start] = 0  # Distance to start is 0
    visited = set()  # Set of visited nodes
    queue = [(0, start)]  # Priority queue (distance, node)
    predecessors = {node: None for node in G.nodes}

    while queue:
        current_distance, current_node = heapq.heappop(queue)

        if current_node == goal:
            return distances[goal], reconstruct_path(start, goal, predecessors), visited

        if current_node in visited:
            continue

        visited.add(current_node)

        for neighbor, weight in successors(current_node):
            new_distance = current_distance + weight
            if new_distance < distances[neighbor]:
                distances[neighbor] = new_distance
                predecessors[neighbor] = current_node
                heapq.heappush(queue, (new_distance, neighbor))

    return float('inf'), []  # No path found


def reconstruct_path(start, goal, predecessors):
    path = [goal]
    while path[-1] != start:
        path.append(predecessors[path[-1]])
    path.reverse()
    return path


start, goal = get_start_goal_nodes()

print(f"{start}, {goal}")

dijkstra_cost, dijkstra_path, visited_nodes = dijkstra(start, goal)
print("Dijkstra Result:")
if dijkstra_cost != float('inf'):
    print(f"Path: {dijkstra_path}")
    print(f"Cost: {dijkstra_cost}\n")
else:
    print("Dijkstra: No solution available for given start and goal\n")

# Convert lines and point coordinates to pyvista graphics
road_network = vistaLines(*ox.graph_to_gdfs(G))
visited_nodes = vistaLines(*ox.graph_to_gdfs(G.subgraph(visited_nodes)))
route_path = vistaLines(
    *ox.graph_to_gdfs(G.subgraph(G.subgraph(dijkstra_path))))
start_coords = pv.PolyData(
    list(list(nodes[nodes['osmid'] == start]['geometry'][start].coords)[0]) + [0])
goal_coords = pv.PolyData(
    list(list(nodes[nodes['osmid'] == goal]['geometry'][goal].coords)[0]) + [0])

# Plot the elevation map
plotter = pv.Plotter()
plotter.add_mesh(road_network, line_width=1,
                 color='blue', label='Road Network')
plotter.add_mesh(visited_nodes, line_width=2,
                 color='green', label='Visited Nodes')
plotter.add_mesh(route_path, line_width=3,
                 color='red', label='Dijkstra Path')
plotter.add_mesh(start_coords, point_size=20,
                 color='black', label='Start')
plotter.add_mesh(goal_coords, point_size=20,
                 color='purple', label='Goal')
plotter.add_legend(bcolor='w', face=None)
plotter.show(cpos='xy')
