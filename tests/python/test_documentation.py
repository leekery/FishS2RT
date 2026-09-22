"""Static checks that keep public documentation and examples executable."""

from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path

import fish_s2

REPOSITORY = Path(__file__).resolve().parents[2]
DOCS = REPOSITORY / "docs"
EXAMPLES = REPOSITORY / "examples"
PYTHON_FENCE = re.compile(r"```python[ \t]*\r?\n(.*?)```", re.DOTALL)
MARKDOWN_LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


class DocumentationTests(unittest.TestCase):
    def test_every_public_symbol_is_in_api_reference(self) -> None:
        reference = (DOCS / "api-reference.md").read_text(encoding="utf-8")
        missing = [name for name in fish_s2.__all__ if name not in reference]
        self.assertEqual([], missing, f"public API symbols missing from docs: {missing}")

    def test_python_fences_parse(self) -> None:
        checked = 0
        for document in sorted(DOCS.glob("*.md")):
            source = document.read_text(encoding="utf-8")
            for index, block in enumerate(PYTHON_FENCE.findall(source), 1):
                with self.subTest(document=document.name, block=index):
                    ast.parse(block, filename=f"{document}#python-{index}")
                checked += 1
        self.assertGreater(checked, 0)

    def test_runnable_examples_compile(self) -> None:
        examples = sorted(EXAMPLES.glob("*.py"))
        self.assertGreater(len(examples), 0)
        for example in examples:
            with self.subTest(example=example.name):
                ast.parse(example.read_text(encoding="utf-8"), filename=str(example))

    def test_local_document_links_exist(self) -> None:
        documents = list(DOCS.glob("*.md")) + [
            REPOSITORY / "README.md",
            REPOSITORY / "README.ru.md",
            EXAMPLES / "README.md",
        ]
        for document in documents:
            source = document.read_text(encoding="utf-8")
            link_base = document.parent
            for raw_target in MARKDOWN_LINK.findall(source):
                target = raw_target.strip().strip("<>")
                if not target or target.startswith(("#", "http://", "https://", "mailto:")):
                    continue
                file_part = target.split("#", 1)[0]
                resolved = (link_base / file_part).resolve()
                with self.subTest(document=document.name, target=target):
                    self.assertTrue(resolved.exists(), f"broken local link: {target}")


if __name__ == "__main__":
    unittest.main()
