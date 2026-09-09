"""Reject incomplete native-probe verdicts; synthetic CPU records only."""
import importlib.util
import json
import hashlib
from pathlib import Path
import random
import os
import shutil
from unittest import mock
from types import SimpleNamespace
import tempfile
import unittest
SPEC = importlib.util.spec_from_file_location('runner', Path(__file__).resolve().parents[1] / '39_run_glm53_probe.py')
runner = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(runner)

class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name); (self.root/'checks').mkdir()
        self.seed=123
        self.binding={'scorer_sha256':'a'*64, 'binary_sha256':'b'*64, 'expected_checks':14,
                      'native_extensions':{name:{'path':'/runtime/'+name+'.so','sha256':'d'*64} for name in ('exllamav3_ext','vllm_exl3_c')}}
        order=runner.native_ids(); random.Random(self.seed).shuffle(order)
        self.rows=[{'native_extension':name,**value} for name,value in self.binding['native_extensions'].items()] + [{'check_order':order}]
        for name in order:
            seed=int.from_bytes(hashlib.sha256(json.dumps([self.seed,name],separators=(',',':')).encode()).digest()[:8],'big') % 2**63
            self.rows.extend([{'event':'start','check_id':name,'global_torch_seed':seed}, {'event':'pass','check_id':name}])
        for i,row in enumerate(self.rows):row['time_unix']=1700000000+i*0.01
        (self.root/'checks/assertion-output.log').write_text('')
        self.seal()

    def seal(self):
        p=self.root/'checks';(p/'raw.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in self.rows))
        (p/'summary.json').write_text(json.dumps({'verdict':'PASS','qualification':'synthetic_native_smoke_only','model_loaded':False,'checks_completed':14,
            'raw_sha256':runner.sha256_file(p/'raw.jsonl'),'test_output_sha256':runner.sha256_file(p/'assertion-output.log')}))
        (p/'manifest.json').write_text(json.dumps({'seed':self.seed,**{k:v for k,v in self.binding.items() if k not in ('native_extensions','cache_layer_types')}}))

    def test_complete_native_record_accepts(self):
        self.assertEqual(runner.score_inner(self.root,'native',self.seed,self.binding)['checks_completed'],14)

    def test_missing_duplicate_reordered_or_failed_native_records_reject(self):
        original=list(self.rows)
        variants=[original[:-1], original+[original[-1]], original[:1]+list(reversed(original[1:])), original+[{'event':'failure'}]]
        for rows in variants:
            self.rows=rows;self.seal()
            with self.assertRaises(ValueError):runner.score_inner(self.root,'native',self.seed,self.binding)

    def test_native_extension_rng_and_schema_mutations_reject(self):
        original=json.loads(json.dumps(self.rows))
        variants=[original[2:], original+[{'arbitrary':'unrecognized'}]]
        for key,value,index in [('path','/unfrozen/fake.so',0),('sha256','e'*64,0),('global_torch_seed',0,3),('time_unix',float('nan'),0)]:
            changed=json.loads(json.dumps(original));changed[index][key]=value;variants.append(changed)
        for rows in variants:
            self.rows=rows;self.seal()
            with self.assertRaises(ValueError):runner.score_inner(self.root,'native',self.seed,self.binding)

    def test_stale_interpreter_or_probe_binding_rejects(self):
        p=self.root/'checks/manifest.json'; original=json.loads(p.read_text())
        for key in ('scorer_sha256','binary_sha256'):
            changed={**original,key:'c'*64};p.write_text(json.dumps(changed))
            with self.assertRaises(ValueError):runner.score_inner(self.root,'native',self.seed,self.binding)

    def test_seed_and_unbound_raw_reject(self):
        with self.assertRaises(ValueError):runner.score_inner(self.root,'native',self.seed+1,self.binding)
        with (self.root/'checks/raw.jsonl').open('a') as f:f.write('{}\n')
        with self.assertRaises(ValueError):runner.score_inner(self.root,'native',self.seed,self.binding)

class CacheRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);(self.root/'checks').mkdir();self.seed=123
        self.binding={'scorer_sha256':'a'*64,'binary_sha256':'b'*64,'cache_layer_types':['deepseek_sparse_attention' if i%4==3 else 'linear_attention' for i in range(45)]}
        order=[f'cache-preflight-{i}' for i in range(5)];random.Random(self.seed).shuffle(order)
        mla=[];tail=[];mamba=[]
        for i,kind in enumerate(self.binding['cache_layer_types']):
            prefix=f'model.language_model.layers.{i}.self_attn'
            if kind=='linear_attention':mamba.append(prefix)
            else:mla += [prefix+'.attn',prefix+'.indexer.k_cache'];tail.append(prefix+'.indexer.tail_cache')
        self.rows=[{'event':'normalized','attention_block_tokens':8704,'mamba_block_tokens':262144,
                    'mamba_shapes':[[3,24576],[64,128,128]],'mamba_dtypes':['torch.bfloat16','torch.float32'],
                    'layout':'LBHNC','shared_pool_blocks':145,'groups':[{'layers':g,'spec':'synthetic'} for g in [mla,tail,mamba[:9],mamba[9:18],mamba[18:26],mamba[26:]]]},
                   {'event':'allocated_and_zeroed','unique_backing_bytes':9565306880,'storage_count':1,'layer_views':67,'cuda_memory_allocated':9565306880,'cuda_memory_reserved':9565306880,'cuda_peak_allocated':9565306880},
                   {'event':'scheduler_normalized','scheduler_block_tokens':4456448,'hash_block_tokens':4456448}]
        for i,name in enumerate(order[:4]):
            ids=list(range(i*36+1,(i+1)*36+1))
            self.rows.append({'event':'reserve','request_id':name,'block_ids':[ids[:31],*[ids[j:j+1] for j in range(31,36)]],'free_blocks':144-36*(i+1)})
        self.rows += [{'event':'fifth_rejected','live_requests':4,'distinct_blocks':144},{'event':'restored','free_blocks':144}]
        for i,row in enumerate(self.rows):row['time_unix']=1700000000+i*0.01
        self.seal()
    def test_page_rounded_storage_is_required(self):
        original = dict(self.rows[1])
        for size in (9565304320, 9565306880 - 4096, 9565306880 + 4096):
            self.rows[1] = {**original, 'unique_backing_bytes': size}; self.seal()
            with self.assertRaisesRegex(ValueError, 'backing'):
                runner.score_inner(self.root, 'cache', self.seed, self.binding)
        self.rows[1] = {**original, 'storage_count': 2}; self.seal()
        with self.assertRaisesRegex(ValueError, 'backing'):
            runner.score_inner(self.root, 'cache', self.seed, self.binding)

    def seal(self):
        p=self.root/'checks';(p/'raw.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in self.rows))
        (p/'manifest.json').write_text(json.dumps({'seed':self.seed,**{k:v for k,v in self.binding.items() if k not in ('native_extensions','cache_layer_types')}}))
        (p/'summary.json').write_text(json.dumps({'verdict':'PASS','qualification':'model_free_cache_allocation_only','actual_input_tokens_processed':0,'model_loaded':False,'raw_sha256':runner.sha256_file(p/'raw.jsonl')}))
    def test_complete_cache_record_accepts(self):
        self.assertEqual(runner.score_inner(self.root,'cache',self.seed,self.binding)['verdict'],'PASS')
    def test_cache_reservations_must_match_frozen_physical_contract(self):
        original=json.loads(json.dumps(self.rows))
        changes=[('request_id','duplicate'),('block_ids',[[1001]]),('free_blocks',999)]
        for key,value in changes:
            self.rows=json.loads(json.dumps(original));self.rows[3][key]=value;self.seal()
            with self.assertRaises(ValueError):runner.score_inner(self.root,'cache',self.seed,self.binding)
    def test_cache_block_reuse_and_seed_order_reject(self):
        self.rows[4]['block_ids']=self.rows[3]['block_ids'];self.seal()
        with self.assertRaises(ValueError):runner.score_inner(self.root,'cache',self.seed,self.binding)

class LaunchBindingTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
    def test_later_beacon_cannot_replace_predetermined_round(self):
        source=Path(__file__).resolve().parents[2]/'results/glm53-flash-gates/native-smoke-003/randomness.json'
        receipt=json.loads(source.read_text());(self.root/'randomness.json').write_text(json.dumps(receipt))
        manifest={'frozen_at_unix':receipt['publication_unix']-45,'node':'/pinned/node','tools':{'/pinned/node':{}},'code':{'scripts/103_verify_drand_receipt_bundle.mjs':{}}}
        with mock.patch.object(runner.subprocess,'run',return_value=SimpleNamespace(returncode=0,stdout='DRAND_BLS_RECEIPT_OK\n',stderr='')):
            with self.assertRaises(ValueError):runner.verify_beacon(self.root,manifest)
    def test_accepted_input_files_must_stay_byte_identical(self):
        for name in ('manifest.json','randomness.json'):(self.root/name).write_text('{}\n')
        accepted={name:runner.sha256_file(self.root/name) for name in ('manifest.json','randomness.json')}
        (self.root/'manifest.json').write_text('{"changed":true}\n')
        with self.assertRaises(ValueError):runner.verify_accepted_inputs(self.root,accepted)
    def test_outer_shell_environment_excludes_ambient_startup_hooks(self):
        with mock.patch.dict(os.environ,{'BASH_ENV':'/bad','LD_PRELOAD':'/bad','PATH':'/bad','PYTHONPATH':'/bad'}):
            environment=runner.wrapper_environment({'GLM_SAFE_PARENT_LOCK_PID':'123'})
        self.assertEqual(environment['PATH'],'/usr/bin:/bin')
        self.assertTrue({'BASH_ENV','LD_PRELOAD','PYTHONPATH'}.isdisjoint(environment))

if __name__=='__main__':unittest.main()

class ComponentLoadRunnerTests(unittest.TestCase):
    def test_load_kinds_and_frozen_dependencies(self):
        for case in ('moe','kda','mla','ordinary'):
            self.assertEqual(runner.probe_verdict('load-'+case,None),'PASS')
            self.assertEqual(runner.probe_verdict('load-'+case,'failed'),'FAIL')
        required={'scripts/42_probe_glm53_load.py','scripts/lib/glm53_load_fixture.py',
                  'scripts/lib/glm53_pinned_stream.py','configs/decision-specs/glm53-load-preflight.json'}
        self.assertTrue(required.issubset(runner.CODE_FILES))

    def test_load_state_rejects_unfrozen_generated_code(self):
        clean={'entries':[{'path':'.','type':'directory'},{'path':'.cache','type':'directory'},
                         {'path':'.humming/tmp/lock/launcher.lock','type':'file','size_bytes':0,'sha256':hashlib.sha256(b'').hexdigest()}]}
        runner.validate_load_state(clean)
        for entry in ({'path':'kernel.cubin','type':'file','size_bytes':1,'sha256':'a'*64},
                      {'path':'library.so','type':'symlink','target':'/tmp/unfrozen'},
                      {'path':'.humming/tmp/lock/launcher.lock','type':'file','size_bytes':1,'sha256':'a'*64}):
            with self.assertRaises(ValueError):runner.validate_load_state({'entries':[entry]})
