"""Real CPU guard witness for named-profile orderly termination; no GPU."""
import hashlib,json,os,signal,subprocess,sys,tempfile,time,unittest
from pathlib import Path
from unittest import mock
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts/lib'))
import glm53_profile as api
class ProfileShutdownTests(unittest.TestCase):
 def test_api_exits_before_guard_and_whole_group_cleanup(self):
  with tempfile.TemporaryDirectory() as directory:
   out=Path(directory);target=out/'server.py';ready=out/'server-ready'
   target.write_text("import signal,time\nfrom pathlib import Path\nstopping=False\ndef stop(*a):\n global stopping\n stopping=True\nsignal.signal(signal.SIGTERM,stop)\nPath("+repr(str(ready))+").touch()\nwhile not stopping:time.sleep(.02)\n")
   guard=subprocess.Popen([sys.executable,'-I','-B',str(ROOT/'scripts/38_guard_glm53_probe.py'),'--output',str(out/'identity'),'--',str(target)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
   child_pid=None
   try:
    deadline=time.monotonic()+10
    while not ready.exists() and time.monotonic()<deadline:time.sleep(.02)
    self.assertTrue(ready.exists());time.sleep(.35)
    row=json.loads((out/'identity/raw.jsonl').read_text().splitlines()[-1]);child_pid=row['pid']
    record={'unit':'cpu-fixture.service','invocation_id':'fixture','control_group':Path('/proc/self/cgroup').read_text().split('::',1)[1].strip(),'controller':api.process_identity(guard.pid),'output':str(out),'ready':True}
    def properties(unit):return {'ActiveState':'active' if guard.poll() is None else 'inactive','InvocationID':'fixture','ControlGroup':record['control_group']}
    forced=[]
    def whole_group(*args,**kwargs):
     forced.append(True);guard.terminate();guard.wait(timeout=5)
     try:os.kill(child_pid,signal.SIGTERM)
     except ProcessLookupError:pass
     return subprocess.CompletedProcess(args,0)
    with mock.patch.object(api,'unit_properties',side_effect=properties),mock.patch.object(api,'check_group_empty'),mock.patch.object(api.subprocess,'run',side_effect=whole_group):
     api.stop_unit(record)
    stdout,stderr=guard.communicate(timeout=8)
    self.assertFalse(forced,'whole-unit stop raced the real guard completion')
    self.assertEqual(guard.returncode,0,stderr)
    summary=json.loads((out/'identity/summary.json').read_text());self.assertEqual(summary['verdict'],'PASS')
    rows=[json.loads(s) for s in (out/'identity/raw.jsonl').read_text().splitlines()]
    self.assertEqual(sum(r.get('completion_verified',False) for r in rows),1)
    self.assertEqual(rows[-1]['event'],'cleanup');self.assertEqual(rows[-1]['live_process_group_after'],[])
   finally:
    if guard.poll() is None:guard.kill()
    guard.communicate(timeout=5)
    if child_pid:
     try:os.kill(child_pid,signal.SIGKILL)
     except ProcessLookupError:pass
 def test_profiles_request_bounded_native_shutdown(self):
  for name in ('agent-fast','1m-experimental'):
   p=json.loads((ROOT/f'configs/profiles/glm-5.3-flash/cuda-spark-128g-{name}.json').read_text());args=p['launch']['args']
   self.assertIn('--shutdown-timeout',args);self.assertEqual(args[args.index('--shutdown-timeout')+1],'10')
if __name__=='__main__':unittest.main()
