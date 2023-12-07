from collections import defaultdict
from heapq import heappop, heappush
import heapq
from typing import List
import json
from graph import Graph
from icecream import ic
import copy

def read_json_data(file_name):
    with open(file_name) as f:
        data = json.load(f)

    data = data["graph"]
    graph = {}
    for node, neighbors in data.items():
        # Create a dictionary for each node
        graph[node] = {}
        for neighbor_node, distance in neighbors.items():
            graph[node][neighbor_node] = distance

    return graph


def dijkstra(graph, source, target):
      """
      Dijkstra's algorithm for finding the shortest path between two nodes in a graph.

      Args:
        graph: A dictionary where the keys are nodes and the values are dictionaries
          mapping other nodes to their edge weights.
        source: The starting node.
        target: The ending node.

      Returns:
        A list of nodes representing the shortest path from source to target,
        or None if no path exists.
      """
      # Initialize distances and predecessors.

      distances = {node: float('inf') for node in graph}
      distances[source] = 0
      predecessors = {node: None for node in graph}

      # Create a priority queue with source as the first element.
      pq = [(0, source)]
      heapq.heapify(pq)

      while pq:
        # Get the node with the smallest distance.
        current_distance, current_node = heapq.heappop(pq)

        # Check if we have reached the target.
        if current_node == target:
          break

        # Update distances of neighbors.
        for neighbor, weight in graph[current_node].items():
          new_distance = distances[current_node] + weight
          if new_distance < distances[neighbor]:
            distances[neighbor] = new_distance
            predecessors[neighbor] = current_node
            heapq.heappush(pq, (new_distance, neighbor))

      # Reconstruct the shortest path.
      path = []
      node = target
      while node is not None:
        path.append(node)
        node = predecessors[node]

      # Reverse the path to get the correct order.
      path.reverse()

      # Check if a path was found.
      if path[-1] != target:
        return None

      return path


def has_edge(graph, start, end):
    if start in graph and end in graph[start]:
        return True
    return False

def add_edge(graph, u, v, w):
    graph[u][v] = w
    graph[v][u] = w

def remove_edge(graph, start, end):
    if start in graph and end in graph[start]:
        del graph[start][end]
        if end in graph and start in graph[end]:
            del graph[end][start]
        return True
    else:
        return False

def find_all_edge(graph, node):
    edge_lst = []

    if node not in graph:
        return []

    for end_node in graph[node]:
        edge_lst.append((node, end_node, graph[node][end_node]))

    return edge_lst


def get_path_length(graph, path):
    length = 0

    if len(path) > 1:
        for i in range(len(path) - 1):
            u = path[i]
            v = path[i + 1]

            length += graph[u][v]

    return length


def yen_k_shortest_paths(graph, source, target, k):
    """
    Implements Yen's algorithm to find k shortest paths from a source node to a target node in a simple graph.

    Args:
        graph: A dictionary where keys are nodes and values are dictionaries mapping neighbors to their weights.
        source: The starting node.
        target: The ending node.
        k: The number of shortest paths to find.

    Returns:
        shortest_paths list of k shortest paths from the source to the target. Each path is represented as a list of nodes.
    """
    shortest_paths = []
    potential_paths = []  # List to store potential kth shortest path
    graph_original = copy.deepcopy(graph)
    # Find the first shortest path
    shortest_paths.append(dijkstra(graph, source, target))
    for i in range(1, k):
        ic(i)
        for j in range(len(shortest_paths[-1]) - 1):
            ic("-----------------------------------------------------")
            spur_node = shortest_paths[i-1][j]
            rootPath = shortest_paths[i-1][:j+1]
            ic(spur_node, rootPath)
            spur_node_plus_one = shortest_paths[i-1][j+1]

            edges_removed = []
            # Remove edges and nodes from the graph that are part of already discovered paths
            for c_path in shortest_paths:
                if rootPath == c_path[:j+1] and (len(c_path) - 1 > j):
                    u = c_path[j]
                    v = c_path[j+1]
                    if has_edge(graph, u, v):
                        edge_weight = graph[u][v]
                        remove_edge(graph, u, v)
                        edges_removed.append((u, v, edge_weight))
            ic(graph)
            # for n in range(len(rootPath) - 1):
            #     node = rootPath[n]
            #     ic(node)
            #     for u, v, weight in find_all_edge(graph, node):
            #
            #         remove_edge(graph, u, v)
            #         edges_removed.append((u, v, weight))


            spur_path = dijkstra(graph, spur_node, target)
            ic(spur_node, spur_path)
            total_path_tempo = rootPath[:-1] + spur_path
            total_path = []
            # removes duplicates nodes
            [total_path.append(item) for item in total_path_tempo if item not in total_path]

            if total_path not in shortest_paths:
                try:
                    if [total_path, get_path_length(graph_original, total_path)] not in potential_paths:
                        potential_paths.append([total_path, get_path_length(graph_original, total_path)])
                except KeyError as e:
                    print("Path not exist")

            for e in edges_removed:
                ic(edges_removed)
                u, v, edge_attr = e
                ic(u, v, edge_attr)
                add_edge(graph, u, v, edge_attr)

            ic("Graph sau khi đã add edge", graph)
            ic(potential_paths)

        min_path = min(potential_paths, key=lambda x: x[1])
        potential_paths.remove(min_path)
        # Attach minimum of potentialSolSpace into shortest_paths dictionary
        shortest_paths.append(min_path[0])
    return (get_path_length(graph, shortest_paths[k-1]), shortest_paths[k-1])

# wiki_graph = {
#     "C": {"D": 3, "E": 2},
#     "D": {"C": 3, "E": 1, "F": 4},
#     "E": {"C": 2, "D": 1, "F": 2, "G": 3},
#     "F": {"D": 4, "E": 2, "G": 2, "H": 1},
#     "G": {"E": 3, "F": 2, "H": 1},
#     "H": {"F": 1, "G": 1}
# }

wiki_graph = {
    "C": {"E": 2, "D": 3},
    "E": {"D": 1, "G": 3, "F": 2},
    "D": {"F": 4},
    "F": {"G": 2 , "H": 1},
    "G": {"H": 2},
    "H": {"G": 2}
}


ggmap = read_json_data("data_name.json")
start = "Ly Thuong Kiet"
end = "Cho Ben Thanh"
#
#
print(dijkstra(ggmap, start, end))
print(yen_k_shortest_paths(ggmap, start, end, 2))
