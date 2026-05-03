"""Causal Dependency Graph - defines execution order for model propagation."""

from dataclasses import dataclass, field


@dataclass
class ModelNode:
    """A node in the causal dependency graph."""
    name: str
    dependencies: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)


CAUSAL_GRAPH = {
    "pricing": ModelNode(
        name="pricing",
        dependencies=[],
        outputs=["predicted_demand", "price_change_pct", "sentiment_impact", "revenue"],
    ),
    "marketing": ModelNode(
        name="marketing",
        dependencies=[],
        outputs=["predicted_conversion", "marketing_effect"],
    ),
    "sentiment": ModelNode(
        name="sentiment",
        dependencies=["pricing", "marketing"],
        outputs=["sentiment_score"],
    ),
    "churn": ModelNode(
        name="churn",
        dependencies=["pricing", "marketing", "sentiment"],
        outputs=["churn_probability"],
    ),
}


def get_execution_order() -> list[str]:
    """Topological sort of the causal graph. Returns model names in execution order."""
    visited = set()
    order = []

    def dfs(node_name: str):
        if node_name in visited:
            return
        visited.add(node_name)
        node = CAUSAL_GRAPH[node_name]
        for dep in node.dependencies:
            dfs(dep)
        order.append(node_name)

    for name in CAUSAL_GRAPH:
        dfs(name)

    return order


def get_propagation_map() -> dict[str, list[str]]:
    """Returns which outputs each model provides to downstream models."""
    return {name: node.outputs for name, node in CAUSAL_GRAPH.items()}
