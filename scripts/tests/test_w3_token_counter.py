#!/usr/bin/env python3
"""Mutation tests for the isolated W3 visible-output token scorer."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCORER = ROOT / "scripts/84_count_glm_output_tokens.py"
TOKENIZER = Path(
    "/home/dsv4/ds4-project/tokenizers/glm52-b4734de4/tokenizer.json"
)
RUNTIME = Path(
    "/home/bmarti44/.cache/glm52-w3-tokenizer-runtime-0.22.2"
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@unittest.skipUnless(TOKENIZER.is_file() and RUNTIME.is_dir(), "Spark tokenizer absent")
class W3TokenCounterTests(unittest.TestCase):
    def invoke(
        self,
        response: Path,
        runtime: Path = RUNTIME,
        cwd: Path | None = None,
        outer_environment: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        init = RUNTIME / "tokenizers/__init__.py"
        native = RUNTIME / "tokenizers/tokenizers.abi3.so"
        command = [
            "/usr/bin/env", "-i",
            "HOME=/nonexistent", "PATH=/usr/bin:/bin",
            "LANG=C.UTF-8", "LC_ALL=C.UTF-8",
            "/usr/bin/python3", "-I", "-B", str(SCORER),
            str(response), str(TOKENIZER), str(runtime),
            digest(TOKENIZER), digest(init), digest(native), "off-warm",
        ]
        return subprocess.run(
            command,
            cwd=cwd,
            env=outer_environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

    @staticmethod
    def response(path: Path, content: str = "x") -> None:
        path.write_text(json.dumps({
            "choices": [{
                "message": {"role": "assistant", "content": content},
                "finish_reason": "length",
            }],
            "usage": {"completion_tokens": 64},
        }), encoding="utf-8")

    def test_fake_cwd_pythonpath_and_sitecustomize_cannot_replace_runtime(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            response = root / "response.json"
            self.response(response)
            (root / "tokenizers.py").write_text(
                "raise RuntimeError('cwd injection loaded')\n", encoding="utf-8"
            )
            fake = root / "fake"
            fake.mkdir()
            (fake / "tokenizers.py").write_text(
                "raise RuntimeError('PYTHONPATH injection loaded')\n", encoding="utf-8"
            )
            (fake / "sitecustomize.py").write_text(
                "raise RuntimeError('sitecustomize injection loaded')\n", encoding="utf-8"
            )
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(fake)
            result = self.invoke(response, cwd=root, outer_environment=environment)
            self.assertEqual(result.returncode, 0, result.stderr)
            record = json.loads(result.stdout)
            self.assertEqual(record["reference_token_count"], 1)
            self.assertEqual(
                record["runtime_init_path"],
                str(RUNTIME / "tokenizers/__init__.py"),
            )

    def test_modified_runtime_dependency_fails_closed(self):
        with tempfile.TemporaryDirectory(
            prefix="glm52-w3-tokenizer-runtime-mutated-",
            dir="/home/bmarti44/.cache",
        ) as temporary:
            runtime = Path(temporary)
            shutil.copytree(RUNTIME / "tokenizers", runtime / "tokenizers")
            init = runtime / "tokenizers/__init__.py"
            init.chmod(0o644)
            init.write_text(init.read_text() + "# mutation\n", encoding="utf-8")
            init.chmod(0o444)
            with tempfile.TemporaryDirectory() as response_directory:
                response = Path(response_directory) / "response.json"
                self.response(response)
                result = self.invoke(response, runtime=runtime)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("digest mismatch", result.stderr)

    def test_ambient_direct_invocation_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            response = Path(temporary) / "response.json"
            self.response(response)
            init = RUNTIME / "tokenizers/__init__.py"
            native = RUNTIME / "tokenizers/tokenizers.abi3.so"
            result = subprocess.run([
                "/usr/bin/python3", str(SCORER), str(response), str(TOKENIZER),
                str(RUNTIME), digest(TOKENIZER), digest(init), digest(native),
                "off-warm",
            ], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
               check=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("closed isolated environment", result.stderr)


if __name__ == "__main__":
    unittest.main()
