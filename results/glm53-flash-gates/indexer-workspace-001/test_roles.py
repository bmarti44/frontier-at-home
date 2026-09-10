"""H1 retarget controls: synthetic manifest, no launch or GPU execution."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('workspace_role_review',HERE/'test_review.py')
review=importlib.util.module_from_spec(spec);spec.loader.exec_module(review)
probe=review.probe

class ExecutedDependencyRoles(unittest.TestCase):
    def test_all_implicit_roles_cannot_be_retargeted_to_bound_node(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);manifest=review.frozen_fixture(root)
            for role in ('safe_wrapper','memory_guard','libc','nvcc','cxx'):
                with self.subTest(role=role):
                    bad=copy.deepcopy(manifest);old=bad['external_dependencies'][role]
                    bad['external_dependencies'][role]=bad['node']
                    bad['files']=[x for x in bad['files'] if x['path']!=old]
                    if role=='nvcc':bad['environment']['DG_JIT_NVCC_COMPILER']=bad['node']
                    with self.assertRaises(ValueError):probe.verify_frozen(root,bad)

    def test_selected_node_cannot_omit_fixed_fetcher_node(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);manifest=review.frozen_fixture(root);old=manifest['node']
            manifest['node']=manifest['external_dependencies']['cxx']
            manifest['files']=[x for x in manifest['files'] if x['path']!=old]
            with self.assertRaises(ValueError):probe.verify_frozen(root,manifest)

    def test_selected_wrapper_cannot_alias_unrelated_bound_node(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);manifest=review.frozen_fixture(root);old=manifest['wrapper']
            manifest['wrapper']=manifest['node']
            manifest['files']=[x for x in manifest['files'] if x['path']!=old]
            with self.assertRaises(ValueError):probe.verify_frozen(root,manifest)

if __name__=='__main__':unittest.main()
