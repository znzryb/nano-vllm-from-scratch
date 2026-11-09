"""Behavior tests use temporary repositories, never the user's configuration."""
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "instruction_generator", Path(__file__).parents[1] / "scripts/generate.py")
gen = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gen)


class GenerationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.write(".ai/INSTRUCTIONS.md", "# Project\n\nShared learning boundary.\n")
        (self.root / ".ai/rules").mkdir()
        self.write(".ai/skills/review/SKILL.md",
                   "---\nname: review\ndescription: Review learning code.\n---\n\n# Review\n")

    def write(self, name, content):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return path

    def add_rule(self, name, patterns, body):
        return self.write(f".ai/rules/{name}.md", "---\npaths:\n"
                          + "".join(f"  - {json.dumps(p)}\n" for p in patterns)
                          + "---\n\n" + body + "\n")

    def run_generator(self, **kwargs):
        with redirect_stdout(io.StringIO()):
            return gen.generate(self.root, **kwargs)

    def tree(self):
        return {p.relative_to(self.root).as_posix(): (p.read_bytes(), p.stat().st_mode)
                for p in self.root.rglob("*") if p.is_file()}

    def test_scopes_are_generated_independently_and_links_relocated(self):
        self.add_rule("code", ["src/engine/**"], "# Code\nUser owns implementation.")
        self.add_rule("deep", ["src/engine/cache/*.py"], "# Cache\nCache invariants.")
        self.add_rule("docs", ["docs/**"], "# Docs\n[Details](../README.md)")
        self.add_rule("glob", ["**/*.cu"], "# CUDA\nGPU evidence.")
        self.run_generator()
        root = (self.root / "AGENTS.md").read_text()
        code = (self.root / "src/engine/AGENTS.md").read_text()
        self.assertIn("GPU evidence.", root)
        self.assertIn("**/*.cu", root)
        self.assertNotIn("User owns implementation.", root)
        self.assertIn("User owns implementation.", code)
        self.assertNotIn("Shared learning boundary.", code)
        self.assertIn("Cache invariants.", (self.root / "src/engine/cache/AGENTS.md").read_text())
        self.assertIn("](../.ai/README.md)", (self.root / "docs/AGENTS.md").read_text())
        skill = (self.root / ".agents/skills/review/SKILL.md").read_text()
        self.assertTrue(skill.startswith("---\nname: review\n"))
        self.assertIn("GENERATED from .ai/skills/review/SKILL.md", skill)
        self.assertFalse((self.root / ".claude").exists())
        self.assertFalse((self.root / ".agents/rules").exists())

    def test_idempotence_and_check_never_writes(self):
        self.run_generator()
        before = self.tree()
        self.assertTrue(self.run_generator(check=True))
        self.run_generator()
        self.assertEqual(before, self.tree())
        self.write(".ai/INSTRUCTIONS.md", "# Changed source\n")
        before = self.tree()
        self.assertFalse(self.run_generator(check=True))
        self.assertEqual(before, self.tree())

    def test_manual_edit_stops_before_any_mutation(self):
        self.add_rule("code", ["src/**"], "# Code\nOriginal.")
        self.run_generator()
        self.write("src/AGENTS.md", "User edited this.\n")
        self.write(".ai/INSTRUCTIONS.md", "# Source changed too\n")
        before = self.tree()
        with self.assertRaises(gen.Conflict):
            self.run_generator()
        self.assertEqual(before, self.tree())

    def test_unknown_native_entries_and_outputs_are_preserved(self):
        for name in ("AGENTS.md", "other/AGENTS.md", "other/AGENTS.override.md",
                     ".agents/skills/review/SKILL.md"):
            with self.subTest(name=name):
                path = self.write(name, "User owned.\n")
                before = self.tree()
                with self.assertRaises(gen.Conflict):
                    self.run_generator()
                self.assertEqual(before, self.tree())
                path.unlink()

    def test_removed_source_only_deletes_owned_unchanged_output(self):
        rule = self.add_rule("code", ["src/**"], "# Code\nBoundary.")
        self.run_generator()
        note = self.write("src/user.txt", "Keep me.\n")
        rule.unlink()
        self.assertFalse(self.run_generator(check=True))
        self.run_generator()
        self.assertFalse((self.root / "src/AGENTS.md").exists())
        self.assertEqual(note.read_text(), "Keep me.\n")

    def test_symlink_sources_and_destinations_are_rejected(self):
        outside = self.root / "outside"
        outside.mkdir()
        self.add_rule("code", ["src/**"], "# Code\nBoundary.")
        (self.root / "src").symlink_to(outside, target_is_directory=True)
        # Unsafe target is represented conditionally at the root, never traversed.
        self.run_generator()
        self.assertFalse((outside / "AGENTS.md").exists())
        self.assertIn("src/**", (self.root / "AGENTS.md").read_text())
        (self.root / ".ai/skills/link").symlink_to(outside, target_is_directory=True)
        with self.assertRaises(gen.Conflict):
            self.run_generator()
        (self.root / ".ai/skills/link").unlink()
        (self.root / "AGENTS.md").unlink()
        (self.root / "AGENTS.md").symlink_to(outside / "missing")
        with self.assertRaises(gen.Conflict):
            self.run_generator()

    def test_submodule_and_nested_git_are_not_written_or_scanned(self):
        self.write(".gitmodules", '[submodule "vendor"]\n path = vendor\n url = example\n')
        self.write("vendor/AGENTS.md", "Upstream policy.\n")
        self.write("nested/.git", "gitdir: elsewhere\n")
        self.write("nested/AGENTS.override.md", "Other repository.\n")
        self.add_rule("vendor", ["vendor/lib/**", "nested/lib/**"], "# External\nRead only.")
        self.run_generator()
        self.assertFalse((self.root / "vendor/lib/AGENTS.md").exists())
        self.assertFalse((self.root / "nested/lib/AGENTS.md").exists())
        self.assertEqual((self.root / "vendor/AGENTS.md").read_text(), "Upstream policy.\n")

    def test_combined_chain_budget_is_checked_before_writing(self):
        self.write(".ai/INSTRUCTIONS.md", "a" * 17000)
        self.add_rule("code", ["src/**"], "b" * 17000)
        with self.assertRaises(gen.Conflict):
            self.run_generator()
        self.assertFalse((self.root / "AGENTS.md").exists())

    def test_unsafe_manifest_and_invalid_rule_fail_before_writing(self):
        self.write(gen.MANIFEST, json.dumps({"version": 1, "files": {"../escape": {}}}))
        with self.assertRaises(gen.Conflict):
            self.run_generator()
        (self.root / gen.MANIFEST).unlink()
        self.write(".ai/rules/bad.md", "---\npaths: [src/**]\n---\nBody.\n")
        with self.assertRaises(gen.Conflict):
            self.run_generator()
        self.assertFalse((self.root / "AGENTS.md").exists())

    def test_skill_permissions_and_resources_are_preserved(self):
        script = self.write(".ai/skills/review/scripts/check.py", "print('ok')\n")
        script.chmod(0o755)
        self.write(".ai/skills/review/references/details.md", "# Details\n")
        self.run_generator()
        target = self.root / ".agents/skills/review/scripts/check.py"
        self.assertEqual(target.read_bytes(), script.read_bytes())
        self.assertEqual(target.stat().st_mode & 0o777, 0o755)
        target.chmod(0o644)
        with self.assertRaises(gen.Conflict):
            self.run_generator()

    def test_write_failure_restores_existing_outputs(self):
        self.run_generator()
        self.write(".ai/INSTRUCTIONS.md", "# Updated source\n")
        self.write(".ai/skills/review/SKILL.md", "# Updated skill\n")
        before = self.tree()
        real_atomic = gen.atomic
        calls = 0

        def fail_second(path, value):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("Simulated disk failure")
            real_atomic(path, value)

        with patch.object(gen, "atomic", side_effect=fail_second):
            with self.assertRaises(OSError):
                self.run_generator()
        self.assertEqual(before, self.tree())


if __name__ == "__main__":
    unittest.main()
