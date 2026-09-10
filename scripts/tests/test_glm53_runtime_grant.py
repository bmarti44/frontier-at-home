"""The optional admin grant permits only fixed runtime control and list calls."""
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
INSTALLER = ROOT / "scripts/dev/grant_glm53_runtime_once.sh"


class RuntimeGrant(unittest.TestCase):
    def policy(self):
        source = INSTALLER.read_text()
        return source.split("<<'POLICY'\n", 1)[1].split("\nPOLICY\n", 1)[0] + "\n"

    def test_real_visudo_accepts_exact_scoped_policy(self):
        policy = self.policy()
        with tempfile.NamedTemporaryFile(mode="w") as f:
            f.write(policy); f.flush()
            result = subprocess.run(["/usr/sbin/visudo", "-cf", f.name], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        rules = [x for x in policy.splitlines() if x and not x.startswith("#")]
        prefix = "bmarti44 ALL=(root) NOPASSWD: "
        self.assertTrue(all(x.startswith(prefix) for x in rules))
        commands = [x.removeprefix(prefix) for x in rules]
        expected = {f"/usr/bin/systemctl {verb} {unit}" for verb in ("start", "stop")
                    for unit in ("docker.service", "docker.socket", "containerd.service")}
        expected.add("/usr/bin/ctr namespaces list -q")
        expected.add("/usr/bin/ctr ^--namespace [A-Za-z0-9_][A-Za-z0-9_.-]* tasks list -q$")
        self.assertEqual(set(commands), expected)
        self.assertEqual(len(commands), len(expected))

    def test_argument_boundary_with_posix_regex_engine(self):
        # grep -E exercises the host's POSIX ERE semantics, not Python's regex dialect.
        policy = self.policy()
        pattern = next(x.split("/usr/bin/ctr ", 1)[1] for x in policy.splitlines() if "/usr/bin/ctr ^" in x)
        for arguments, allowed in [
            ("--namespace moby tasks list -q", True),
            ("--namespace k8s.io tasks list -q", True),
            ("--namespace test_1-a tasks list -q", True),
            ("--namespace --address=/tmp/socket tasks list -q", False),
            ("--namespace moby tasks exec -q", False),
            ("--namespace moby tasks list -q --address=/tmp/socket", False),
            ("--namespace a b tasks list -q", False),
            ("--namespace ../../tmp tasks list -q", False),
            ("--namespace moby tasks list -q; /bin/sh", False),
            ("--namespace moby tasks list -q\n/bin/sh", False),
        ]:
            with self.subTest(arguments=arguments):
                # NUL records keep embedded newlines inside the argument string.
                result = subprocess.run(["/usr/bin/grep", "-zEx", pattern], input=arguments + "\0",
                                        capture_output=True, text=True)
                matches = result.returncode == 0
                self.assertEqual(matches, allowed)

    def test_no_effect_before_root_and_no_service_actions_in_installer(self):
        source = INSTALLER.read_text()
        result = subprocess.run(["/usr/bin/bash", str(INSTALLER)], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("one-time", result.stderr)
        # The only systemctl/ctr occurrences belong to the policy heredoc.
        outside = source.replace(self.policy().rstrip("\n"), "")
        self.assertNotRegex(outside, r"/(?:usr/)?(?:s?bin)/(?:systemctl|ctr)\b")
        self.assertNotIn("NOPASSWD: ALL", source)


if __name__ == "__main__":
    unittest.main()
