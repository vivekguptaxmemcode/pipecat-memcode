from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

EXAMPLE_PATH = (
    Path(__file__).resolve().parents[1] / "examples" / "foundational" / "memcode_memory.py"
)


def test_foundational_example_gracefully_drains_normal_disconnect():
    tree = ast.parse(EXAMPLE_PATH.read_text(encoding="utf-8"))
    handlers = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "on_client_disconnected"
    ]

    assert len(handlers) == 1
    calls = [call for call in ast.walk(handlers[0]) if isinstance(call, ast.Call)]
    called_methods = {call.func.attr for call in calls if isinstance(call.func, ast.Attribute)}
    awaited_runner_stop = any(
        isinstance(node, ast.Await)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "stop_when_done"
        and isinstance(node.value.func.value, ast.Name)
        and node.value.func.value.id == "runner"
        for node in ast.walk(handlers[0])
    )

    assert awaited_runner_stop
    assert "cancel" not in called_methods
    assert "end" not in called_methods


def test_foundational_example_exposes_local_oauth_disconnect_command():
    source = EXAMPLE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = {node.name for node in ast.walk(tree) if isinstance(node, ast.AsyncFunctionDef)}

    assert "disconnect_account" in functions
    assert 'sys.argv[1] == "--disconnect"' in source


def test_foundational_example_imports_with_advertised_extra(monkeypatch):
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    spec = importlib.util.spec_from_file_location("pipecat_memcode_example_smoke", EXAMPLE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)
