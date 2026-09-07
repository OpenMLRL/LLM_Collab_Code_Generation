"""Ensure every training launcher isolates its cache before doing work."""

import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class RuntimeCacheEntrypointTests(unittest.TestCase):
    def test_all_training_mains_configure_cache_first(self):
        if (ROOT / "single_turn").is_dir():
            paths = sorted(ROOT.glob("single_turn/train/train_*.py"))
            expected = 3  # MAGRPO, shared MAPL, centralized; wrappers delegate.
        elif (ROOT / "native_parallel").is_dir():
            paths = sorted(ROOT.glob("*/train/train_*.py"))
            expected = 8
        else:
            paths = sorted(ROOT.glob("train_*.py"))
            expected = 9
        checked = 0
        for path in paths:
            tree = ast.parse(path.read_text())
            for node in tree.body:
                if not isinstance(node, ast.FunctionDef) or node.name != "main":
                    continue
                with self.subTest(entrypoint=str(path.relative_to(ROOT))):
                    body = node.body[1:] if ast.get_docstring(node) is not None else node.body
                    first, second = body[:2]
                    self.assertIsInstance(first, ast.ImportFrom)
                    self.assertEqual(first.module, "comlrl.runtime")
                    self.assertEqual(first.names[0].name, "configure_job_cuda_cache")
                    self.assertIsInstance(second, ast.Expr)
                    self.assertIsInstance(second.value, ast.Call)
                    self.assertEqual(second.value.func.id, "configure_job_cuda_cache")
                checked += 1
        self.assertEqual(checked, expected)


if __name__ == "__main__":
    unittest.main()
