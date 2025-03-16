import networkx as nx
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

PARENT_ID_COL = "parent_id"
CAT_ID_COL = "cat_id"


class CategoryTree:
    def __init__(self, category_tree_path: str):
        self.category_tree = pd.read_csv(category_tree_path, usecols=[PARENT_ID_COL, CAT_ID_COL])

        self._root_node = 0
        self.category_tree = self.category_tree.replace(np.nan, self._root_node)
        self.category_tree = self.category_tree.astype(int)
        edges = self.category_tree[[PARENT_ID_COL, CAT_ID_COL]].to_records(index=False)

        self._category_tree_graph = nx.DiGraph()
        self._category_tree_graph.add_edges_from(edges)

        self._leaf_nodes = [
            node
            for node in self._category_tree_graph.nodes()
            if self._category_tree_graph.out_degree(node) == 0
            and self._category_tree_graph.in_degree(node) == 1
        ]

        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self._leaf_nodes)

    @property
    def leaf_nodes(self) -> list[int]:
        return self._leaf_nodes

    @property
    def inverted_edge_dict(self) -> dict[int | None, int]:
        pairs = zip(
            self.category_tree[CAT_ID_COL], self.category_tree[PARENT_ID_COL], strict=False
        )
        pairs = [
            (source, parent if parent is not self._root_node else np.NaN)
            for source, parent in pairs
        ]
        return dict(pairs)
