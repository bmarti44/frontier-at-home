"""Fixed acceptance for optional named GLM profiles; no model is loaded."""
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/lib'))
import profile_resolver as resolver

class ExperimentalProfiles(unittest.TestCase):
    def test_profiles_render_exact_geometry_and_containment(self):
        for name, cap, batch, kv in [('agent-fast', 65536, 512, 4294967296),
                                     ('1m-experimental', 262144, 128, 9565304320)]:
            p = resolver.load_profile('glm-5.3-flash', 'cuda-spark-128g-' + name + '.json')
            host = resolver.load_host(ROOT / 'configs/hosts/spark-aba1.json')
            d = resolver.resolve(p, resolver.load_model('glm-5.3-flash'), host, run_root='/tmp/glm-test')
            self.assertEqual(d['status'], 'estimated')
            self.assertIsNone(d['switch_alias'])
            self.assertEqual(d['context_cap'], cap * 4)
            for flag, value in [('--max-model-len', cap), ('--max-num-seqs', 4),
                                ('--max-num-batched-tokens', batch), ('--kv-cache-memory-bytes', kv)]:
                self.assertEqual(d['argv'][d['argv'].index(flag)+1], str(value))
            self.assertEqual(d['port'], 8015)
            self.assertEqual(d['env']['HOME'], '/tmp/glm-test/state')
            self.assertNotIn('CUDA_LAUNCH_BLOCKING', d['env'])
            self.assertEqual(d['systemd']['properties']['MemoryMax'], '94G')
            self.assertEqual(d['systemd']['properties']['MemorySwapMax'], '0')
            self.assertEqual(d['safety']['minimum_start_gib'], 110)
            self.assertEqual(d['safety']['kill_floor_gib'], 18)
            self.assertGreaterEqual(len(d['digest_checks']), 3)

    def test_profile_entry_rejects_production_and_cli_overrides(self):
        api = importlib.import_module('glm53_profile')
        with self.assertRaisesRegex(ValueError, 'experimental'):
            api.resolve_profile('glm-5.3-flash/cuda-spark-128g-1m', None, Path('/tmp/test'))
        with self.assertRaisesRegex(ValueError, 'override'):
            api.reject_overrides(['--profile', 'test', '--prefill-batch', '2048'])

    def test_frozen_inventory_rejects_changed_or_unlisted_files(self):
        api = importlib.import_module('glm53_profile')
        import hashlib
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'runtime'; root.mkdir()
            (root/'a').write_bytes(b'a')
            manifest = Path(tmp)/'inventory.json'
            manifest.write_text(json.dumps({'files':[{'path':'a','size_bytes':1,'sha256':hashlib.sha256(b'a').hexdigest(),'identity':[]}]}))
            api.verify_frozen_tree(root, manifest)
            (root/'a').write_bytes(b'b')
            with self.assertRaises(ValueError): api.verify_frozen_tree(root, manifest)
            (root/'a').write_bytes(b'a'); (root/'extra').write_text('extra')
            with self.assertRaises(ValueError): api.verify_frozen_tree(root, manifest)

    def test_stale_process_record_cannot_stop_a_process(self):
        api = importlib.import_module('glm53_profile')
        import os
        with self.assertRaisesRegex(ValueError, 'identity'):
            api.open_identity({'pid':os.getpid(),'start_ticks':-1,'command':['not this process']})


class LifecycleRegression(unittest.TestCase):
    def setUp(self):
        from unittest import mock
        self.mock = mock
        self.api = importlib.import_module('glm53_profile')

    def test_failed_readiness_stops_unit_and_waits_for_controller(self):
        api=self.api
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);state=out/'state.json'
            child=self.mock.Mock(pid=os.getpid());child.poll.return_value=None
            props={'ActiveState':'active','InvocationID':'fresh','ControlGroup':'/fake'}
            snapshot={'profile_id':'glm-5.3-flash/cuda-spark-128g-agent-fast','safety':{'startup_timeout_seconds':10}}
            with self.mock.patch.object(api,'lifecycle_path',return_value=state), \
                 self.mock.patch.object(api.subprocess,'Popen',return_value=child), \
                 self.mock.patch.object(api,'unit_properties',return_value=props), \
                 self.mock.patch.object(api,'authenticated_ready',side_effect=ValueError('wrong model')), \
                 self.mock.patch.object(api,'stop_unit') as stop:
                with self.assertRaisesRegex(ValueError,'wrong model'):
                    api.run_contained(['wrapper','--tag','glm-test'],{},out,snapshot)
                stop.assert_called_once()
                self.assertEqual(stop.call_args.args[0]['invocation_id'],'fresh')
                child.wait.assert_called_once_with(timeout=60)
                self.assertFalse(json.loads(state.read_text())['ready'])

    def test_term_reaches_unit_and_waits_for_descendants(self):
        import signal
        api=self.api
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);state=out/'state.json';child=self.mock.Mock(pid=os.getpid(),returncode=0)
            child.poll.side_effect=[None,None,0]
            snapshot={'profile_id':'glm-5.3-flash/cuda-spark-128g-agent-fast','safety':{'startup_timeout_seconds':10},'port':8015}
            props={'ActiveState':'active','InvocationID':'fresh','ControlGroup':'/fake'}
            with self.mock.patch.object(api,'lifecycle_path',return_value=state), \
                 self.mock.patch.object(api.subprocess,'Popen',return_value=child), \
                 self.mock.patch.object(api,'unit_properties',return_value=props), \
                 self.mock.patch.object(api,'authenticated_ready',return_value={'pass':True}), \
                 self.mock.patch.object(api.time,'sleep',side_effect=lambda _:signal.raise_signal(signal.SIGTERM)), \
                 self.mock.patch.object(api,'stop_unit') as stop:
                self.assertEqual(api.run_contained(['wrapper','--tag','glm-test'],{},out,snapshot),0)
                self.assertGreaterEqual(stop.call_count,1)
                child.wait.assert_called_once_with(timeout=60)

    def test_concurrent_start_cannot_replace_live_record(self):
        import fcntl
        api=self.api
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);state=out/'state.json';state.write_text('{"existing":"record"}')
            before=state.read_bytes()
            with state.with_suffix('.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                with self.mock.patch.object(api,'lifecycle_path',return_value=state), self.mock.patch.object(api.subprocess,'Popen') as spawn:
                    with self.assertRaisesRegex(ValueError,'active launcher'):
                        api.run_contained([],{},out,{'profile_id':'test'})
                    spawn.assert_not_called()
                    self.assertEqual(state.read_bytes(),before)

    def test_unit_reuse_cannot_be_stopped(self):
        api=self.api
        with self.mock.patch.object(api,'unit_properties',return_value={'ActiveState':'active','InvocationID':'new'}), \
             self.mock.patch.object(api.subprocess,'run') as run:
            with self.assertRaisesRegex(ValueError,'invocation identity'):
                api.stop_unit({'unit':'old.service','invocation_id':'old'})
            run.assert_not_called()

    def test_containment_drift_rejected(self):
        api=self.api
        p=resolver.load_profile('glm-5.3-flash','cuda-spark-128g-agent-fast.json')
        p['containment']['memory_max']='100G'
        with self.mock.patch.object(api.resolver,'load_profile',return_value=p):
            with self.assertRaisesRegex(ValueError,'containment'):
                api.resolve_profile('glm-5.3-flash/cuda-spark-128g-agent-fast',ROOT/'configs/hosts/spark-aba1.json',Path('/tmp/test'))

    def test_readiness_requires_auth_model_and_completed_semantics(self):
        import http.server
        import threading
        api=self.api
        mode={'value':'good'}
        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self,*args): pass
            def do_GET(self):
                if self.path=='/health': self.send_response(200);self.end_headers();return
                if self.headers.get('Authorization')!='Bearer '+'a'*32 and mode['value']!='no-auth':
                    self.send_response(401);self.end_headers();return
                self.send_response(200);self.end_headers()
                self.wfile.write(json.dumps({'data':[{'id':'wrong' if mode['value']=='wrong-model' else 'glm-5.3-flash'}]}).encode())
            def do_POST(self):
                self.rfile.read(int(self.headers['Content-Length']))
                self.send_response(200);self.end_headers()
                self.wfile.write(json.dumps({'choices':[{'finish_reason':'length' if mode['value']=='short' else 'stop','message':{'content':'READY'}}]}).encode())
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);(out/'api-key').write_text('a'*32)
            server=http.server.ThreadingHTTPServer(('127.0.0.1',0),Handler)
            worker=threading.Thread(target=server.serve_forever);worker.start()
            try:
                snapshot={'port':server.server_port}
                self.assertTrue(api.authenticated_ready(snapshot,out)['semantic_completion'])
                for value in ('no-auth','wrong-model','short'):
                    mode['value']=value
                    with self.subTest(value=value),self.assertRaises(ValueError):
                        api.authenticated_ready(snapshot,out)
            finally: server.shutdown();server.server_close();worker.join()



class ReviewRegression(unittest.TestCase):
    def setUp(self):
        from unittest import mock
        self.mock=mock;self.api=importlib.import_module('glm53_profile')

    def test_unknown_unit_observation_raises(self):
        with self.mock.patch.object(self.api.subprocess,'run',return_value=self.mock.Mock(returncode=1,stdout='',stderr='bus failed')):
            with self.assertRaisesRegex(ValueError,'observe'):
                self.api.unit_properties('test.service')

    def test_terminal_unit_still_checks_descendants(self):
        api=self.api
        with self.mock.patch.object(api,'unit_properties',return_value={'ActiveState':'failed','InvocationID':'same'}), self.mock.patch.object(api,'check_group_empty',create=True,side_effect=ValueError('descendants')) as check:
            with self.assertRaisesRegex(ValueError,'descendants'):
                api.stop_unit({'unit':'test.service','invocation_id':'same','control_group':'/fake'})
            check.assert_called_once()

    def test_stop_error_cannot_skip_wait_or_restore_handlers(self):
        import signal
        api=self.api;before=signal.getsignal(signal.SIGTERM)
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);child=self.mock.Mock(pid=os.getpid());child.poll.return_value=None
            with self.mock.patch.object(api,'lifecycle_path',return_value=out/'state.json'), self.mock.patch.object(api.subprocess,'Popen',return_value=child), self.mock.patch.object(api,'unit_properties',return_value={'ActiveState':'active','InvocationID':'fresh','ControlGroup':'/fake'}), self.mock.patch.object(api,'authenticated_ready',side_effect=ValueError('wrong model')), self.mock.patch.object(api,'stop_unit',side_effect=TimeoutError('stop timeout')):
                try:
                    with self.assertRaises(TimeoutError):
                        api.run_contained(['wrapper','--tag','test'],{},out,{'profile_id':'test','safety':{'startup_timeout_seconds':10}})
                    child.wait.assert_called_once_with(timeout=60)
                    self.assertIs(signal.getsignal(signal.SIGTERM),before)
                finally: signal.signal(signal.SIGTERM,before)

    def test_preparation_registers_and_serializes_before_hashing(self):
        api=self.api
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);path=out/'state.json'
            with self.mock.patch.object(api,'lifecycle_path',return_value=path):
                with api.preparing_session({'profile_id':'test'},out):
                    record=json.loads(path.read_text())
                    self.assertEqual(record['phase'],'preparing')
                    fd=api.open_identity(record['launcher'])
                    import os;os.close(fd)
                    with self.assertRaisesRegex(ValueError,'active launcher'):
                        with api.preparing_session({'profile_id':'test'},out): pass



    def test_real_preparing_process_is_cancelled_before_model_launch(self):
        import subprocess,time
        api=self.api
        with tempfile.TemporaryDirectory() as tmp:
            state=Path(tmp)/'state.json'
            code = """import sys,time
from pathlib import Path
sys.path.insert(0,sys.argv[1])
import glm53_profile as p
p.lifecycle_path=lambda name:Path(sys.argv[2])
with p.preparing_session({'profile_id':'glm-5.3-flash/cuda-spark-128g-agent-fast'},Path(sys.argv[2]).parent):
 print('preparing',flush=True)
 time.sleep(30)
 raise RuntimeError('uncancelled preparation')
"""
            child=subprocess.Popen([sys.executable,'-B','-c',code,str(ROOT/'scripts/lib'),str(state)],stdout=subprocess.PIPE,text=True)
            try:
                self.assertEqual(child.stdout.readline().strip(),'preparing')
                with self.mock.patch.object(api,'lifecycle_path',return_value=state):
                    api.lifecycle_action('glm-5.3-flash/cuda-spark-128g-agent-fast','stop')
                self.assertEqual(child.wait(timeout=5),143)
                record=json.loads(state.read_text())
                self.assertEqual(record['phase'],'exited');self.assertIsNone(record['unit'])
            finally:
                if child.poll() is None:child.terminate();child.wait(timeout=5)
                child.stdout.close()

if __name__ == '__main__': unittest.main()
