"""Incremental storage union cannot duplicate shared memory or alias weights. CPU only."""
import copy
import importlib.util
from pathlib import Path
import unittest

class GrowthTests(unittest.TestCase):
    def setUp(self):
        spec=importlib.util.spec_from_file_location('growth_probe',Path(__file__).resolve().parents[1]/'43_probe_glm53_growth.py')
        self.api=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.api)

    def test_required_storage_union_and_overlap(self):
        self.assertEqual(self.api.minimum_bytes(1),2124585984)
        self.assertEqual(self.api.minimum_bytes(2),3947169792)
        self.assertEqual(self.api.minimum_bytes(2)-self.api.minimum_bytes(1),1822583808)
        for bad in (0,3,True):
            with self.assertRaises(ValueError):self.api.minimum_bytes(bad)

    def test_shared_backing_identity_must_match(self):
        first={'shared_scratch':[{'storage_pointer':100,'storage_bytes':64}],
               'tensor_cache':[{'storage_pointer':200,'storage_bytes':16}]}
        second=copy.deepcopy(first)
        self.api.require_shared_identity(first,second)
        for key in ('shared_scratch','tensor_cache'):
            changed=copy.deepcopy(second);changed[key][0]['storage_pointer']+=1
            with self.assertRaises(ValueError):self.api.require_shared_identity(first,changed)
        changed=copy.deepcopy(second);changed['shared_scratch']=[]
        with self.assertRaises(ValueError):self.api.require_shared_identity(first,changed)

    def test_parameter_and_table_storage_are_independent(self):
        self.api.require_disjoint([{'storage_pointer':100,'storage_bytes':64}],
                                  [{'storage_pointer':200,'storage_bytes':64}])
        for pointer in (100,110,150):
            with self.assertRaises(ValueError):self.api.require_disjoint(
                [{'storage_pointer':100,'storage_bytes':64}], [{'storage_pointer':pointer,'storage_bytes':64}])

if __name__=='__main__':unittest.main()
