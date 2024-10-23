from algorithms.structures.node import NodeBase

from typing import Optional


def generate_graph(self, root: NodeBase, path: str = "mcts_tree") -> None:
    dot = graphviz.Digraph(comment='MCTS Tree')

    def add_node_to_graph(node: NodeBase, parent: Optional[NodeBase] = None, edge_label: Optional[str] = None) -> None:
        node_label = f"""State: {node.state}\nVisits: {
            node.visits}\nReward: {node.reward:.2f}"
        node_id = f"node_{node.uid}"""
        dot.node(node_id, label=node_label)

        if parent is not None:
            parent_id = f"node_{parent.uid}"
            dot.edge(parent_id, node_id, label=edge_label)

        for child in node.children:
            child_label = f"Action: {child.action}"
            add_node_to_graph(child, child.parent, child_label)

    # Add root node and recursively add all children
    add_node_to_graph(root)

    dot.render(path, format="png", cleanup=True)
    print(f"Tree graph generated and saved as '{path}.png'")
