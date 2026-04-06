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
    """Yield public module-level functions and class methods, skipping nested local functions."""

    for stmt in module_node.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not stmt.name.startswith("_"):
                yield stmt
        elif isinstance(stmt, ast.ClassDef):
            for class_stmt in stmt.body:
                if isinstance(class_stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if not class_stmt.name.startswith("_"):
                        yield class_stmt


class SourceDocstringContractTestCase(unittest.TestCase):
    def test_source_function_documentation(self):
        source_dir = Path(__file__).resolve().parents[1] / "source"
        total_count = 0
        passed_count = 0
        current_file = None

        print("\n")  # Add newline to separate from unittest verbose output

        for py_file in sorted(source_dir.glob("*.py")):
            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
            file_label = py_file.relative_to(source_dir.parent)

            # Print file header when switching files
            if current_file != file_label:
                if current_file is not None:
                    print()  # Blank line between files
                print(f"{file_label}:")
                current_file = file_label

            for node in _iter_api_functions(tree):
                total_count += 1
                func_label = f"{file_label}:{node.lineno} {node.name}"

                with self.subTest(function=func_label):
                    doc = ast.get_docstring(node) or ""
                    documented_params = set(PARAM_RE.findall(doc))
                    documented_types = set(TYPE_RE.findall(doc))
                    issues = []

                    for param in _function_params(node):
                        if param not in documented_params:
                            issues.append(f"missing ':param {param}:'")
                        if param not in documented_types:
                            issues.append(f"missing ':type {param}:'")

                    if _has_direct_value_return(node):
                        if ":return:" not in doc:
                            issues.append("missing ':return:'")
                        if ":rtype:" not in doc:
                            issues.append("missing ':rtype:'")

                    if len(issues) == 0:
                        passed_count += 1
                        print(f"  ✓ {node.lineno} {node.name}")
                    else:
                        print(f"  ✗ {node.lineno} {node.name}: {' | '.join(issues)}")

                    self.assertEqual(
                        len(issues),
                        0,
                        msg=" | ".join(issues) if issues else "",
                    )

        print(f"\n{passed_count}/{total_count} functions passed docstring requirements")


if __name__ == "__main__":
    unittest.main()
