"""Load probe byte coverage and storage metadata; synthetic CPU fixtures only."""
import importlib.util
from pathlib import Path
import unittest
import copy


class LoadProbeTests(unittest.TestCase):
    def setUp(self):
        spec=importlib.util.spec_from_file_location('load_probe_test',Path(__file__).resolve().parents[1]/'42_probe_glm53_load.py')
        self.api=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.api)

    def test_constructor_parameter_formulas(self):
        expected={'moe':1822563072,'kda':53707440,'mla':4214800,'ordinary':1268776960}
        for case,total in expected.items():
            self.assertEqual(self.api.layout_bytes(self.api.parameter_layout(case)),total)
        with self.assertRaises(ValueError):self.api.parameter_layout('full-model')

    def test_storage_description_keeps_alias_and_offset(self):
        import torch
        base=torch.arange(30,dtype=torch.int16).reshape(5,6);view=base[:,2:4]
        a,b=self.api.describe(base),self.api.describe(view)
        self.assertEqual(a['storage_pointer'],b['storage_pointer'])
        self.assertEqual(b['data_pointer']-a['data_pointer'],4)
        self.assertEqual(b['shape'],[5,2]);self.assertEqual(b['stride'],[6,1])
        self.assertEqual(b['storage_bytes'],60)

    def test_byte_coverage_rejects_missing_duplicate_mismatched_or_bad_transfer(self):
        names={'tensor-a':'a'*64,'tensor-b':'b'*64}
        rows=[]
        for name,digest in names.items():
            rows.extend([{'event':'loaded','name':name,'sha256':digest},
                         {'event':'transfer','name':name,'source_sha256':digest,'device_sha256':digest},
                         {'event':'final_bytes','name':name,'sha256':digest}])
        self.api.validate_byte_coverage(rows,names)
        variants=[rows[:-1],rows+[rows[0]],copy.deepcopy(rows),copy.deepcopy(rows)]
        variants[2][0]['sha256']='0'*64;variants[3][1]['device_sha256']='0'*64
        for changed in variants:
            with self.assertRaises(ValueError):self.api.validate_byte_coverage(changed,names)


if __name__=='__main__':unittest.main()
