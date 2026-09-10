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
   child_pid=None;child_fd=None
   try:
    deadline=time.monotonic()+10
    while not ready.exists() and time.monotonic()<deadline:time.sleep(.02)
    self.assertTrue(ready.exists());time.sleep(.35)
    row=json.loads((out/'identity/raw.jsonl').read_text().splitlines()[-1]);child_pid=row['pid'];child_fd=os.pidfd_open(child_pid)
    record={'unit':'cpu-fixture.service','invocation_id':'fixture','control_group':Path('/proc/self/cgroup').read_text().split('::',1)[1].strip(),'controller':api.process_identity(guard.pid),'output':str(out),'ready':True}
    def properties(unit):return {'ActiveState':'active' if guard.poll() is None else 'inactive','InvocationID':'fixture','ControlGroup':record['control_group']}
    forced=[]
    def whole_group(*args,**kwargs):
     forced.append(True);guard.terminate();guard.wait(timeout=5)
     try:signal.pidfd_send_signal(child_fd,signal.SIGTERM)
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
    if child_fd is not None:
     try:signal.pidfd_send_signal(child_fd,signal.SIGKILL)
     except ProcessLookupError:pass
     os.close(child_fd)
 def test_profiles_request_bounded_native_shutdown(self):
  for name in ('agent-fast','1m-experimental'):
   p=json.loads((ROOT/f'configs/profiles/glm-5.3-flash/cuda-spark-128g-{name}.json').read_text());args=p['launch']['args']
   self.assertIn('--shutdown-timeout',args);self.assertEqual(args[args.index('--shutdown-timeout')+1],'10')
 def test_unverified_api_identity_is_never_signaled(self):
  current=api.process_identity(os.getpid());group=Path('/proc/self/cgroup').read_text()
  baseline={'argv':current['command'],'executable':current['executable'],'cgroup':group,'binary_sha256':api.sha256_file(Path('/proc/self/exe'))}
  row={'event':'identity','pid':os.getpid(),'start_ticks':current['start_ticks'],'cgroup':group,'executable_verified':True,'argv_verified':True,'environment_verified':True}
  for field in ('start_ticks','argv','executable','cgroup','binary_sha256','argv_verified'):
   with self.subTest(field=field),tempfile.TemporaryDirectory() as directory:
    out=Path(directory);(out/'identity').mkdir();manifest=dict(baseline);sample=dict(row)
    if field=='start_ticks':sample[field]=-1
    elif field=='argv_verified':sample[field]=False
    else:manifest[field]=[] if field=='argv' else 'wrong'
    (out/'identity/manifest.json').write_text(json.dumps(manifest));(out/'identity/raw.jsonl').write_text(json.dumps(sample)+'\n')
    record={'unit':'fixture','invocation_id':'fixture','output':str(out),'control_group':group.split('::',1)[1].strip()}
    with mock.patch.object(api,'unit_properties',return_value={'ActiveState':'active','InvocationID':'fixture','ControlGroup':record['control_group']}),mock.patch.object(api.signal,'pidfd_send_signal') as send:
     with self.assertRaises((ValueError,FileNotFoundError)):api.orderly_api_stop(record)
     send.assert_not_called()
 def test_timeout_uses_group_cleanup_and_cannot_report_clean_shutdown(self):
  record={'unit':'fixture','invocation_id':'fixture','ready':True}
  active={'ActiveState':'active','InvocationID':'fixture'};inactive={'ActiveState':'inactive'}
  with mock.patch.object(api,'unit_properties',side_effect=[active,active,inactive]),mock.patch.object(api,'orderly_api_stop',side_effect=TimeoutError('expired')),mock.patch.object(api,'check_group_empty'),mock.patch.object(api,'verify_guard_completion'),mock.patch.object(api.subprocess,'run') as stop:
   api.stop_unit(record)
   stop.assert_called_once_with(['systemctl','--user','stop','fixture'],check=True,timeout=60)
   self.assertFalse(record['shutdown']['clean']);self.assertIn('expired',record['shutdown']['failure'])
if __name__=='__main__':unittest.main()
