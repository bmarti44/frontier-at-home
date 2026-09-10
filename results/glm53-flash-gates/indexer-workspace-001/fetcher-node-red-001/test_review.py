"""H1/H2 regressions: synthetic CPU evidence, no native/control execution.

The H2 tests execute the actual controller's post-host-score join statements and
actual inner scorer on complete full-geometry fixture bytes. Only the already
closed host scorer is replaced by an explicitly synthetic verified-host result.
"""
import ast
import copy
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('workspace_review_test',HERE/'probe.py')
probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)


def binding(path):
    return {'path':str(path),'size_bytes':path.stat().st_size,'sha256':probe.sha(path)}


def frozen_fixture(root):
    for name in probe.REQUIRED_CODE:
        p=root/'code'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(name)
    (root/'metadata').mkdir();(root/'cache-template').mkdir()
    (root/'metadata/config.json').write_text('{}');(root/'cache-template-inventory.json').write_text('{}')
    runtime=root/'external/runtime';(runtime/'bin').mkdir(parents=True)
    python=runtime/'bin/python3';python.write_text('synthetic interpreter bytes; never executed')
    # Read actual small execution dependencies only; these are never executed.
    node=root/'external/node';node.write_text('synthetic node; never executed')
    paths={'node':str(node),'wrapper':str(probe.REPO/'results/glm52-gates/harness/glm_cgroup_run.sh'),
        'safe_wrapper':str(probe.REPO/'results/glm52-gates/harness/glm_safe_run.sh'),
        'memory_guard':str(probe.REPO/'scripts/03_memory_guard.py'),
        'libc':'/usr/lib/aarch64-linux-gnu/libc.so.6','nvcc':'/usr/local/cuda-13.0/bin/nvcc','cxx':'/usr/bin/c++'}
    inventory=root/'metadata/runtime-inventory.json'
    probe.write(inventory,{'schema_version':1,'files':[{**binding(python),'path':'bin/python3'}]})
    return {'files':[binding(p) for p in root.rglob('*') if p.is_file()]+[binding(Path(v)) for k,v in paths.items() if k!='node'],
        'safety':copy.deepcopy(probe.SAFETY),'cases':copy.deepcopy(probe.CASES),'budgets_mib':[512,64],
        'python':str(python),'node':paths['node'],'wrapper':paths['wrapper'],
        'runtime':{'root':str(runtime),'inventory':str(inventory),'verified_files':1},
        'external_dependencies':{k:paths[k] for k in ('safe_wrapper','memory_guard','libc','nvcc','cxx')},
        'environment':{'DG_JIT_NVCC_COMPILER':paths['nvcc'],'CUDA_HOME':'/usr/local/cuda-13.0','PATH':f'{runtime}/bin:/usr/local/cuda-13.0/bin:/usr/bin:/bin'}}


class ExternalBindings(unittest.TestCase):
    def test_complete_selected_dependencies_accept(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);probe.verify_frozen(root,frozen_fixture(root))

    def test_each_selected_external_binding_is_mandatory(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);manifest=frozen_fixture(root)
            selected=[manifest[k] for k in ('python','node','wrapper')]+list(manifest['external_dependencies'].values())
            for path in selected:
                with self.subTest(path=path):
                    bad=copy.deepcopy(manifest);bad['files']=[x for x in bad['files'] if x['path']!=path]
                    with self.assertRaises(ValueError):probe.verify_frozen(root,bad)

    def test_retargeted_selected_launch_paths_reject(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);manifest=frozen_fixture(root)
            for role in ('python','node','wrapper'):
                with self.subTest(role=role):
                    bad=copy.deepcopy(manifest);bad[role]=str(root/'never-frozen'/role)
                    with self.assertRaises(ValueError):probe.verify_frozen(root,bad)

    def test_python_must_belong_to_verified_runtime(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);manifest=frozen_fixture(root)
            rogue=root/'external/second-python';rogue.write_text('also hash bound, but not in runtime')
            bad=copy.deepcopy(manifest);bad['python']=str(rogue);bad['files'].append(binding(rogue))
            with self.assertRaises(ValueError):probe.verify_frozen(root,bad)

    def test_selected_inventory_must_be_normalized_frozen_inventory(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);manifest=frozen_fixture(root)
            wrong=root/'external/other-inventory.json';wrong.write_text('{}')
            bad=copy.deepcopy(manifest);bad['runtime']['inventory']=str(wrong);bad['files'].append(binding(wrong))
            with self.assertRaises(ValueError):probe.verify_frozen(root,bad)

    def test_frozen_interpreter_bytes_match_inventory_entry(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);manifest=frozen_fixture(root);inventory=Path(manifest['runtime']['inventory'])
            data=probe.read_json(inventory);data['files'][0]['sha256']='0'*64;probe.write(inventory,data)
            manifest['files']=[binding(inventory) if x['path']==str(inventory) else x for x in manifest['files']]
            with self.assertRaises(ValueError):probe.verify_frozen(root,manifest)

    def test_missing_declared_dependency_role_rejects(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);manifest=frozen_fixture(root)
            for role in manifest['external_dependencies']:
                with self.subTest(role=role):
                    bad=copy.deepcopy(manifest);del bad['external_dependencies'][role]
                    with self.assertRaises(ValueError):probe.verify_frozen(root,bad)

    def test_compiler_environment_must_match_frozen_role(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);manifest=frozen_fixture(root)
            other=root/'external/second-nvcc';other.write_text('different compiler; also bound')
            manifest['files'].append(binding(other));manifest['environment']['DG_JIT_NVCC_COMPILER']=str(other)
            with self.assertRaises(ValueError):probe.verify_frozen(root,manifest)


def make_actual_inner(root):
    """Complete CPU oracle artifacts; synthetic memory values are not measurements."""
    import numpy as np
    seed=42;case='prefill-4';budget=512;cfg=probe.fixture.case_config(seed,case)
    indices=np.full((512,2048),-1,dtype='<i4')
    # Reuse each request's true top-history vector, retaining the original oracle.
    pools={}
    for i,(request,position,old) in enumerate(zip(cfg['row_requests'],cfg['positions'],cfg['history_counts'])):
        key=(int(request),int(old))
        if key not in pools:pools[key]=probe.fixture.top_history(seed,*key)[:511]
        indices[i,:2044]=(pools[key][:,None]*4+np.arange(4)).reshape(-1)
        count=(int(position)+1)%4;indices[i,2044:2044+count]=np.arange(int(position)//4*4,int(position)//4*4+count)
    cache,tail=probe.fixture.cache_and_tail(seed,case,final=True)
    for name,array in [('indices.i32.gz',indices),('cache.u8.gz',cache),('tail.bf16.gz',tail)]:
        with gzip.open(root/name,'wb',compresslevel=1) as f:f.write(array.tobytes())
    with gzip.open(root/'logits.f32.gz','wb',compresslevel=1) as f:
        for request,position,old in zip(cfg['row_requests'],cfg['positions'],cfg['history_counts']):
            f.write(probe.fixture.valid_logits(seed,int(request),int(old),(int(position)+1)//4).tobytes())
    memory={'cuda_allocated':probe.BASE_BYTES+1024,'cuda_reserved':probe.BASE_BYTES+512*1024**2+4096,
        'cuda_peak_allocated':probe.BASE_BYTES+512*1024**2+2048,'cuda_peak_reserved':probe.BASE_BYTES+512*1024**2+4096,
        'device_free':100*1024**3,'device_total':120*1024**3,'workspace_bytes':probe.WORKSPACE}
    rows=[{'time_unix':110,'event':'configured',**probe.geometry(seed,case,budget)},
        {'time_unix':120,'event':'profiled','workspace_bytes':probe.WORKSPACE,'cuda_peak_allocated':memory['cuda_peak_allocated'],'cuda_reserved':memory['cuda_reserved'],'device_total':memory['device_total']},
        {'time_unix':130,'event':'start','case':case},
        {'time_unix':140,'event':'output','case':case,'budget_mib':budget,'artifacts':{n:probe.sha(root/n) for n in probe.artifact_sizes(seed,case)},
         'calls':[{**probe.call_specs(case,budget)[0],'cuda_allocated_at_return':memory['cuda_peak_allocated']}],
         'metadata':probe.expected_metadata(seed,case,budget),'memory':memory,'cuda_elapsed_ms':1,'cache_stride':[8448,132,1],
         'tail_stride':[1024,512,128,1],'output_alias':True,'input_digests':probe.expected_inputs(seed,case)}]
    (root/'traceback.log').write_text('');probe.write(root/'manifest.json',{});probe.write(root/'summary.json',{})
    return rows


class IdentityInterval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(prefix='glm53-workspace-h2-cpu-');cls.armroot=Path(cls.temp.name)
        cls.checks=cls.armroot/'checks';cls.checks.mkdir();cls.rows=make_actual_inner(cls.checks)
        identities=[{'event':'identity','time_unix':100,'pid':123,'start_ticks':456,'completion_verified':False,'terminal_exec_filter_verified':False},
                    {'event':'identity','time_unix':200,'pid':123,'start_ticks':456,'completion_verified':True,'terminal_exec_filter_verified':True},
                    {'event':'cleanup','time_unix':201,'live_process_group_after':[]}]
        (cls.armroot/'identity').mkdir();raw=''.join(json.dumps(x)+'\n' for x in identities);(cls.armroot/'identity/raw.jsonl').write_text(raw)
        probe.write(cls.armroot/'identity/summary.json',{'verdict':'PASS','raw_sha256':hashlib.sha256(raw.encode()).hexdigest(),'identity_samples':2,
            'probe_exit_code':0,'failure':None,'live_process_group_after':[],'qualification':'Python_probe_identity_only'})
        cls.host={'verdict':'PASS','qualification':'host_and_probe_identity_observations_only','identity_pid':123,'identity_start_ticks':456,'identity_samples':2}
        tree=ast.parse((HERE/'run.py').read_text());execute=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='execute')
        first=next(i for i,n in enumerate(execute.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='host_result' for t in n.targets))
        last=next(i for i,n in enumerate(execute.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='cache' for t in n.targets))
        cls.join_code=compile(ast.Module(body=execute.body[first:last],type_ignores=[]),str(HERE/'run.py'),'exec')

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def run_join(self,times):
        rows=copy.deepcopy(self.rows)
        for row,t in zip(rows,times):row['time_unix']=t
        (self.checks/'raw.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
        # This is the actual numerical/evidence scorer, with no patched oracle.
        probe.write(self.checks/'summary.json',probe.score_arm(self.checks,42,'prefill-4',512))
        arm={'case':'prefill-4','arm':'baseline-a','budget_mib':512};bindings={'manifest.json':'a'*64,'randomness.json':'b'*64};source=HERE/'probe.py'
        probe.write(self.checks/'manifest.json',{'seed':42,**arm,'source_sha256':probe.sha(source),'frozen_manifest_sha256':bindings['manifest.json'],'randomness_sha256':bindings['randomness.json']})
        env={'armroot':self.armroot,'expected':{},'score_host_observations':lambda root,expected:self.host,
             'probe':probe,'strict_json':probe.read_json,'require':probe.require,'seed':42,'arm':arm,'source':source,
             'sha256_file':probe.sha,'bindings':bindings,'Path':Path,'json':json,'hashlib':hashlib,'write':lambda path,data:None}
        exec(self.join_code,env)

    def test_valid_inner_interval_accepts(self):self.run_join([110,120,130,140])
    def test_inner_before_verified_initial_identity_rejects(self):
        with self.assertRaises(ValueError):self.run_join([1,2,3,4])
    def test_inner_after_verified_completion_identity_rejects(self):
        with self.assertRaises(ValueError):self.run_join([201,202,203,204])
    def test_only_one_output_past_completion_rejects(self):
        with self.assertRaises(ValueError):self.run_join([110,120,130,200.0001])


if __name__=='__main__':unittest.main()
