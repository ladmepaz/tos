from __future__ import annotations

from typing import Any

import networkx as nx
from networkx.algorithms.community.louvain import louvain_communities

YEAR = "year"
LEAF = "leaf"
ROOT = "root"
TRUNK = "trunk"
BRANCH = "branch"
SAP = "_sap"
LEAF_CONNECTIONS = "_leaf_connections"
ELABORATE_SAP = "_elaborate_sap"
ROOT_CONNECTIONS = "_root_connections"
RAW_SAP = "_raw_sap"
MIN_LEAF_CONNECTIONS = 3
MAX_LEAF_AGE_YEARS = 7


def _limit(attribute: list[tuple[Any, int | float]], _max: int) -> list[tuple[Any, int | float]]:
    if _max is not None:
        sorted_attribute = sorted(attribute, key=lambda x: (x[1], str(x[0])), reverse=True)
        attribute = sorted_attribute[:_max]
    return attribute


class Sap:
    """Baseline SAP algorithm replicated from bibx."""

    def __init__(
        self,
        max_roots: int = 20,
        max_leaves: int = 50,
        max_trunk: int = 20,
        max_branch_size: int = 15,
    ) -> None:
        self.max_roots = max_roots
        self.max_leaves = max_leaves
        self.max_trunk = max_trunk
        self.max_branch_size = max_branch_size
        self.min_leaf_connections = MIN_LEAF_CONNECTIONS
        self.max_leaf_age = MAX_LEAF_AGE_YEARS

    @staticmethod
    def clean_graph(graph: nx.DiGraph) -> nx.DiGraph:
        """Match bibx clean_graph behavior."""
        giant_component_nodes = max(nx.weakly_connected_components(graph), key=len)
        giant = graph.subgraph(giant_component_nodes).copy()

        giant.remove_nodes_from(
            [n for n in giant if giant.in_degree(n) == 1 and giant.out_degree(n) == 0]
        )

        loops = [loop for loop in nx.strongly_connected_components(giant) if len(loop) > 1]
        for loop in loops:
            giant.remove_edges_from([(u, v) for u in loop for v in loop])

        return giant

    def tree(self, graph: nx.DiGraph) -> nx.DiGraph:
        graph = graph.copy()
        graph = self._compute_root(graph)
        graph = self._compute_leaves(graph)
        graph = self._compute_sap(graph)
        graph = self._compute_trunk(graph)
        return self._compute_branches(graph)

    def _compute_root(self, graph: nx.DiGraph) -> nx.DiGraph:
        g = graph.copy()
        valid_roots = [(n, g.in_degree(n)) for n in g.nodes if g.out_degree(n) == 0]
        sorted_roots = _limit(valid_roots, self.max_roots)
        nx.set_node_attributes(g, 0, ROOT)
        for node, degree in sorted_roots:
            g.nodes[node][ROOT] = degree
        return g

    def _compute_leaves(self, graph: nx.DiGraph) -> nx.DiGraph:
        g = graph.copy()
        roots = [n for n, d in g.nodes.items() if d[ROOT] > 0]
        if not roots:
            raise TypeError("It's necessary to have some roots")

        nx.set_node_attributes(g, 0, ROOT_CONNECTIONS)
        for node in roots:
            g.nodes[node][ROOT_CONNECTIONS] = 1

        topological_order = list(nx.topological_sort(g))
        for node in reversed(topological_order):
            neighbors = list(g.successors(node))
            if neighbors:
                g.nodes[node][ROOT_CONNECTIONS] = sum(
                    g.nodes[n][ROOT_CONNECTIONS] for n in neighbors
                )

        potential_leaves = [
            (node, g.nodes[node][ROOT_CONNECTIONS]) for node in g.nodes if g.in_degree(node) == 0
        ]
        extended_leaves = potential_leaves[:]

        if self.min_leaf_connections is not None:
            potential_leaves = [(n, c) for n, c in potential_leaves if c >= self.min_leaf_connections]

        if self.max_leaf_age is not None:
            potential_leaves = [
                (n, c)
                for n, c in potential_leaves
                if YEAR in g.nodes[n] and g.nodes[n][YEAR] is not None
            ]
            newest_year = max(g.nodes[n][YEAR] for n, _ in potential_leaves if g.nodes[n][YEAR])
            earliest_publication_year = newest_year - self.max_leaf_age
            potential_leaves = [
                (n, c)
                for n, c in potential_leaves
                if g.nodes[n][YEAR] >= earliest_publication_year
            ]

        if not potential_leaves:
            potential_leaves = extended_leaves

        potential_leaves = _limit(potential_leaves, self.max_leaves)
        nx.set_node_attributes(g, 0, LEAF)
        for node, connections in potential_leaves:
            g.nodes[node][LEAF] = connections
        return g

    @staticmethod
    def _raw_sap(graph: nx.DiGraph) -> nx.DiGraph:
        g = graph.copy()
        valid_root = [n for n, d in g.nodes.items() if d[ROOT] > 0]
        if not valid_root:
            raise TypeError("The graph needs to have at least some roots")

        nx.set_node_attributes(g, 0, ROOT_CONNECTIONS)
        nx.set_node_attributes(g, 0, RAW_SAP)

        for node in valid_root:
            g.nodes[node][RAW_SAP] = g.nodes[node][ROOT]
            g.nodes[node][ROOT_CONNECTIONS] = 1

        for node in reversed(list(nx.topological_sort(g))):
            neighbors = list(g.successors(node))
            if not neighbors:
                continue
            for attr in (RAW_SAP, ROOT_CONNECTIONS):
                g.nodes[node][attr] = sum(g.nodes[neighbor][attr] for neighbor in neighbors)

        return g

    @staticmethod
    def _elaborate_sap(graph: nx.DiGraph) -> nx.DiGraph:
        g = Sap._raw_sap(graph)
        valid_leaf = [n for n, d in g.nodes.items() if d[LEAF] > 0]
        if not valid_leaf:
            raise TypeError("The graph needs to have at least some leaves")

        nx.set_node_attributes(g, 0, ELABORATE_SAP)
        nx.set_node_attributes(g, 0, LEAF_CONNECTIONS)
        for node in valid_leaf:
            g.nodes[node][ELABORATE_SAP] = g.nodes[node][LEAF]
            g.nodes[node][LEAF_CONNECTIONS] = 1

        for node in nx.topological_sort(g):
            neighbors = list(g.predecessors(node))
            if neighbors:
                for attr in (ELABORATE_SAP, LEAF_CONNECTIONS):
                    g.nodes[node][attr] = sum(g.nodes[neighbor][attr] for neighbor in neighbors)

        return g

    @staticmethod
    def _compute_sap(graph: nx.DiGraph) -> nx.DiGraph:
        g = Sap._raw_sap(graph)
        g = Sap._elaborate_sap(g)

        nx.set_node_attributes(g, 0, SAP)
        for node in g.nodes:
            g.nodes[node][SAP] = (
                g.nodes[node][LEAF_CONNECTIONS] * g.nodes[node][RAW_SAP]
                + g.nodes[node][ROOT_CONNECTIONS] * g.nodes[node][ELABORATE_SAP]
            )

        return g

    def _compute_trunk(self, graph: nx.DiGraph) -> nx.DiGraph:
        g = graph.copy()
        potential_trunk = [
            (n, d[SAP])
            for n, d in g.nodes.items()
            if d[ROOT] == 0 and d[LEAF] == 0 and d[SAP] > 0
        ]
        if not potential_trunk:
            raise TypeError("The graph needs to have at least some nodes with sap")

        potential_trunk = _limit(potential_trunk, self.max_trunk)
        nx.set_node_attributes(g, 0, TRUNK)
        for node, sap_value in potential_trunk:
            g.nodes[node][TRUNK] = sap_value
        return g

    def _compute_branches(self, graph: nx.DiGraph) -> nx.DiGraph:
        g = graph.copy()
        undirected = g.to_undirected()
        communities = louvain_communities(undirected, seed=0)
        branches = sorted(communities, key=lambda community: (len(community), sorted(community)[0]))[:3]
        nx.set_node_attributes(g, 0, BRANCH)

        for branch_index, branch in enumerate(branches, start=1):
            potential_branch = [
                (n, g.nodes[n][YEAR])
                for n in branch
                if g.nodes[n][ROOT] == 0
                and g.nodes[n][TRUNK] == 0
                and g.nodes[n][YEAR] is not None
            ]
            potential_branch = _limit(potential_branch, self.max_branch_size)
            for node, _ in potential_branch:
                g.nodes[node][BRANCH] = branch_index

        return g
