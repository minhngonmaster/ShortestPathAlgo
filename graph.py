

class Graph:
    def __init__(self):
        self.nodes = {}

    def add_nodes(self, node):
        self.nodes[node] = {}

    def add_edge(self, start, end, distance):
        self.nodes[start][end] = distance

    def has_edge(self, start, end):
        if start in self.nodes and end in self.nodes[start]:
            return True
        return False

    def remove_edge(self, start, end):
        if start in self.nodes and end in self.nodes[start]:
            del self.nodes[start][end]
            return True
        else:
            return False

if __name__ == "__main__":
    graph = Graph()
    # Add nodes and edges

    graph.add_nodes("A")
    graph.add_nodes("B")
    graph.add_edge("A", "B", 1000)
    graph.add_edge("A", "C", 2000)
    graph.remove_edge("A", "B")
    print(graph.has_edge("A", "B"))

    print(graph.nodes)
