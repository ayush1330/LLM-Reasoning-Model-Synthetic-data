"""
Logic tree computation engine.

Handles topological ordering, safe expression evaluation, and rule application
to compute ground truth values from structured tax law cases.
"""

import ast
from typing import Any, Dict, List

from .models import Case, Node, NodeType, Rule, RuleKind


class LogicTree:
    """Computes ground truth values from structured tax law cases."""

    def __init__(self, case: Case):
        """Initialize with a tax case."""
        self.case = case
        self.computed_values: Dict[str, Any] = {}

    def compute(self) -> Dict[str, Any]:
        """
        Compute all node values in topological order.

        Returns:
            Dict mapping node_id to computed value
        """
        # Get nodes in topological order
        ordered_nodes = self._topological_sort()

        # Compute each node in order
        for node_id in ordered_nodes:
            node = self.case.nodes[node_id]
            value = self._compute_node(node)
            self.computed_values[node_id] = value

        return self.computed_values

    def _topological_sort(self) -> List[str]:
        """
        Sort nodes in topological order based on dependencies.

        Returns:
            List of node_ids in computation order
        """
        visited = set()
        temp_visited = set()
        result = []

        def visit(node_id: str):
            if node_id in temp_visited:
                raise ValueError(f"Circular dependency detected involving {node_id}")
            if node_id in visited:
                return

            temp_visited.add(node_id)

            # Visit dependencies first
            node = self.case.nodes[node_id]
            for dep in node.depends_on:
                visit(dep)

            temp_visited.remove(node_id)
            visited.add(node_id)
            result.append(node_id)

        # Visit all nodes
        for node_id in self.case.nodes:
            if node_id not in visited:
                visit(node_id)

        return result

    def _compute_node(self, node: Node) -> Any:
        """Compute value for a single node."""
        if node.node_type == NodeType.FACT:
            # Facts come from the case facts
            fact = self.case.facts.get(node.node_id)
            return fact.value if fact else None

        elif node.node_type == NodeType.DERIVED:
            # Derived nodes use formulas or rules
            if node.formula:
                # Special handling for deduction_choice to track path
                if node.node_id == "deduction_choice":
                    return self._compute_deduction_choice()
                else:
                    return self._safe_eval(node.formula, self.computed_values)
            elif node.rule_id:
                rule = self.case.rules[node.rule_id]
                return self._apply_rule(rule, self.computed_values)

        elif node.node_type == NodeType.OUTPUT:
            # Output nodes are computed like derived nodes
            if node.formula:
                return self._safe_eval(node.formula, self.computed_values)
            elif node.rule_id:
                rule = self.case.rules[node.rule_id]
                return self._apply_rule(rule, self.computed_values)

        return None

    def _compute_deduction_choice(self) -> Any:
        """Compute deduction choice and track which path was chosen."""
        standard_amount = self.computed_values.get("standard_deduction_amount", 0)
        itemized_amount = self.computed_values.get("itemized_deduction", 0)

        # Determine which deduction is higher
        if standard_amount >= itemized_amount:
            # Store the path choice in computed_values for later access
            self.computed_values["deduction_path"] = "standard"
            return standard_amount
        else:
            self.computed_values["deduction_path"] = "itemized"
            return itemized_amount

    def _safe_eval(self, expression: str, context: Dict[str, Any]) -> Any:
        """
        Safely evaluate mathematical expressions.

        Args:
            expression: Mathematical expression string
            context: Variable context for evaluation

        Returns:
            Evaluated result
        """
        # Parse the expression into an AST
        try:
            tree = ast.parse(expression, mode='eval')
        except SyntaxError:
            raise ValueError(f"Invalid expression syntax: {expression}")

        # Evaluate the AST safely
        return self._eval_ast(tree.body, context)

    def _eval_ast(self, node: ast.AST, context: Dict[str, Any]) -> Any:
        """Safely evaluate AST nodes."""
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Name):
            if node.id in context:
                return context[node.id]
            else:
                raise ValueError(f"Undefined variable: {node.id}")
        elif isinstance(node, ast.BinOp):
            left = self._eval_ast(node.left, context)
            right = self._eval_ast(node.right, context)

            # Handle None values
            if left is None:
                left = 0
            if right is None:
                right = 0

            if isinstance(node.op, ast.Add):
                return left + right
            elif isinstance(node.op, ast.Sub):
                return left - right
            elif isinstance(node.op, ast.Mult):
                return left * right
            elif isinstance(node.op, ast.Div):
                if right == 0:
                    raise ValueError("Division by zero")
                return left / right
            else:
                raise ValueError(f"Unsupported operator: {type(node.op)}")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id == 'min':
                    if len(node.args) != 2:
                        raise ValueError("min() requires exactly 2 arguments")
                    left = self._eval_ast(node.args[0], context)
                    right = self._eval_ast(node.args[1], context)
                    # Handle None values
                    if left is None:
                        left = 0
                    if right is None:
                        right = 0
                    return min(left, right)
                elif node.func.id == 'max':
                    if len(node.args) != 2:
                        raise ValueError("max() requires exactly 2 arguments")
                    left = self._eval_ast(node.args[0], context)
                    right = self._eval_ast(node.args[1], context)
                    # Handle None values
                    if left is None:
                        left = 0
                    if right is None:
                        right = 0
                    return max(left, right)
                else:
                    raise ValueError(f"Unsupported function: {node.func.id}")
            else:
                raise ValueError("Unsupported function call")
        elif isinstance(node, ast.Compare):
            # Handle comparison operations (==, !=, <, >, <=, >=)
            if len(node.ops) != 1 or len(node.comparators) != 1:
                raise ValueError("Only single comparisons are supported")

            left = self._eval_ast(node.left, context)
            right = self._eval_ast(node.comparators[0], context)
            op = node.ops[0]

            if isinstance(op, ast.Eq):
                return left == right
            elif isinstance(op, ast.NotEq):
                return left != right
            elif isinstance(op, ast.Lt):
                return left < right
            elif isinstance(op, ast.LtE):
                return left <= right
            elif isinstance(op, ast.Gt):
                return left > right
            elif isinstance(op, ast.GtE):
                return left >= right
            else:
                raise ValueError(f"Unsupported comparison operator: {type(op)}")
        elif isinstance(node, ast.IfExp):
            # Handle ternary operator: x if condition else y
            condition = self._eval_ast(node.test, context)
            true_value = self._eval_ast(node.body, context)
            false_value = self._eval_ast(node.orelse, context)
            return true_value if condition else false_value
        else:
            raise ValueError(f"Unsupported AST node type: {type(node)}")

    def _apply_rule(self, rule: Rule, context: Dict[str, Any]) -> Any:
        """
        Apply a tax rule to compute a value.

        Args:
            rule: Rule to apply
            context: Current computation context

        Returns:
            Computed value
        """
        if rule.kind == RuleKind.CAP:
            # For cap rules, evaluate the formula which should contain min()
            return self._safe_eval(rule.formula, context)
        elif rule.kind == RuleKind.PHASEOUT:
            # For phaseout rules, evaluate the formula which should contain if() and max()
            return self._safe_eval(rule.formula, context)
        else:
            # For other rule kinds, just evaluate the formula
            return self._safe_eval(rule.formula, context)
