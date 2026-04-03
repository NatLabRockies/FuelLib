import ast
import re
import unittest
from pathlib import Path


PARAM_RE = re.compile(r":param\s+([A-Za-z_][A-Za-z0-9_]*)\s*:")
TYPE_RE = re.compile(r":type\s+([A-Za-z_][A-Za-z0-9_]*)\s*:")


class _DirectReturnVisitor(ast.NodeVisitor):
    """Detect value returns while ignoring nested function/lambda scopes."""

    def __init__(self):
        self.has_value_return = False

    def visit_Return(self, node):
        if node.value is not None and not (
            isinstance(node.value, ast.Constant) and node.value.value is None
        ):
            self.has_value_return = True

    def visit_FunctionDef(self, node):
        return

    def visit_AsyncFunctionDef(self, node):
        return

    def visit_Lambda(self, node):
        return


def _function_params(node):
    params = []
    for arg in node.args.posonlyargs + node.args.args + node.args.kwonlyargs:
        if arg.arg not in {"self", "cls"}:
            params.append(arg.arg)

    if node.args.vararg and node.args.vararg.arg not in {"self", "cls"}:
        params.append(node.args.vararg.arg)

    if node.args.kwarg and node.args.kwarg.arg not in {"self", "cls"}:
        params.append(node.args.kwarg.arg)

    return params


def _has_direct_value_return(node):
    visitor = _DirectReturnVisitor()
    for stmt in node.body:
        visitor.visit(stmt)
    return visitor.has_value_return


def _iter_api_functions(module_node):
    """Yield module-level functions and class methods, skipping nested local functions."""

    for stmt in module_node.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield stmt
        elif isinstance(stmt, ast.ClassDef):
            for class_stmt in stmt.body:
                if isinstance(class_stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    yield class_stmt


class SourceDocstringContractTestCase(unittest.TestCase):
    def test_source_functions_document_inputs_and_outputs(self):
        source_dir = Path(__file__).resolve().parents[1] / "source"
        issues = []

        for py_file in sorted(source_dir.glob("*.py")):
            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))

            for node in _iter_api_functions(tree):

                doc = ast.get_docstring(node) or ""
                documented_params = set(PARAM_RE.findall(doc))
                documented_types = set(TYPE_RE.findall(doc))

                for param in _function_params(node):
                    if param not in documented_params:
                        issues.append(
                            f"{py_file.relative_to(source_dir.parent)}:{node.lineno} "
                            f"{node.name} missing ':param {param}:'"
                        )
                    if param not in documented_types:
                        issues.append(
                            f"{py_file.relative_to(source_dir.parent)}:{node.lineno} "
                            f"{node.name} missing ':type {param}:'"
                        )

                if _has_direct_value_return(node):
                    if ":return:" not in doc:
                        issues.append(
                            f"{py_file.relative_to(source_dir.parent)}:{node.lineno} "
                            f"{node.name} missing ':return:'"
                        )
                    if ":rtype:" not in doc:
                        issues.append(
                            f"{py_file.relative_to(source_dir.parent)}:{node.lineno} "
                            f"{node.name} missing ':rtype:'"
                        )

        if issues:
            self.fail(
                "Source docstring contract violations found:\n"
                + "\n".join(f"- {issue}" for issue in issues)
            )


if __name__ == "__main__":
    unittest.main()
