"""Hard rule: library code imports only numpy and the standard library."""
import ast
import pathlib
import sys

LIB = pathlib.Path(__file__).resolve().parent.parent / "tensorgrad"


def test_library_imports_only_numpy_and_stdlib():
    for path in LIB.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                names = [node.module]
            else:
                continue
            for name in names:
                top = name.split(".")[0]
                assert top == "numpy" or top in sys.stdlib_module_names, f"{path.name}: {name}"
