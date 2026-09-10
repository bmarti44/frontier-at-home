"""CPU-only pre-CUDA admission tests of the actual guarded serve() function."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

SOURCE=Path(__file__).resolve().parents[3]/'scripts/47_run_glm53_dev.py'
spec=importlib.util.spec_from_file_location('file_cache_target',SOURCE);api=importlib.util.module_from_spec(spec);spec.loader.exec_module(api)
FLAG='GLM53_RELEASE_MODEL_FILE_CACHE'

class FileCacheAdmissionTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name);self.model=self.root/'model';self.model.mkdir();self.server=self.root/'server';self.server.mkdir()
  self.payload=self.model/'weights.safetensors';self.payload.write_bytes(b'CPU regression fixture only')
  self.inventory={'files':[{'path':self.payload.name,'size_bytes':self.payload.stat().st_size,'sha256':api.sha(self.payload)}]}
  self.launch={'arguments':['--model',str(self.model)],'environment':{FLAG:'1'}}
  self.save_inventory();(self.server/'api-key').write_text('synthetic-test-key')
  self.events=[];self.descriptors=[]
 def save_inventory(self):
  p=self.model/'inventory.json';p.write_text(json.dumps(self.inventory));self.launch['model_inventory']={'sha256':api.sha(p)};self.save_launch()
 def save_launch(self):(self.server/'launch.json').write_text(json.dumps(self.launch))
 def advice(self,fd,offset,length,mode):
  self.assertEqual((offset,length,mode),(0,0,os.POSIX_FADV_DONTNEED));self.assertEqual(os.fstat(fd).st_ino,self.payload.stat().st_ino);self.descriptors.append(fd);self.events.append('advice')
 def run_serve(self,engine=None,advice=None):
  with mock.patch.object(api.runpy,'run_module',side_effect=engine or (lambda *a,**k:self.events.append('engine'))),mock.patch.object(api.os,'posix_fadvise',side_effect=advice or self.advice),mock.patch.object(sys,'path',list(sys.path)),mock.patch.object(sys,'argv',list(sys.argv)),mock.patch.dict(os.environ,{},clear=False),mock.patch('builtins.print'):
   api.serve(self.server)
 def test_enabled_releases_pages_before_engine_import(self):
  def engine(*a,**k):
   if self.events!=['advice']:raise MemoryError('model page cache was not advised before CUDA admission')
   self.events.append('engine')
  self.run_serve(engine);self.assertEqual(self.events,['advice','engine'])
  for fd in self.descriptors:
   with self.assertRaises(OSError):os.fstat(fd)
 def test_default_and_explicit_off_skip_advice(self):
  for setting in [None,'0']:
   self.events=[];self.launch['environment']={} if setting is None else {FLAG:setting};self.save_launch();self.run_serve(advice=lambda *a: self.fail('disabled advice'));self.assertEqual(self.events,['engine'])
 def test_changed_inventory_rejects_before_engine(self):
  (self.model/'inventory.json').write_text('{}')
  with self.assertRaises(ValueError):self.run_serve()
  self.assertEqual(self.events,[])
 def test_escaping_symlink_and_wrong_size_reject(self):
  for mode in ['escape','symlink','size']:
   with self.subTest(mode=mode):
    original=dict(self.inventory['files'][0]);self.events=[]
    if mode=='escape':self.inventory['files'][0]['path']='../weights.safetensors'
    elif mode=='symlink':
     (self.model/'link').symlink_to(self.payload);self.inventory['files'][0]['path']='link'
    else:self.inventory['files'][0]['size_bytes']+=1
    self.save_inventory()
    with self.assertRaises((ValueError,OSError)):self.run_serve()
    self.assertEqual(self.events,[]);self.inventory['files'][0]=original
 def test_advice_failure_aborts_and_closes_descriptor(self):
  def fail(fd,*args):self.descriptors.append(fd);raise OSError('synthetic advice failure')
  with self.assertRaises(OSError):self.run_serve(advice=fail)
  self.assertEqual(self.events,[])
  for fd in self.descriptors:
   with self.assertRaises(OSError):os.fstat(fd)
 def test_payload_bytes_unchanged_and_not_rehashed(self):
  original=self.payload.read_bytes();real_sha=api.sha;hashed=[]
  def digest(p):hashed.append(Path(p));return real_sha(p)
  with mock.patch.object(api,'sha',side_effect=digest):self.run_serve()
  self.assertEqual(hashed,[self.model/'inventory.json']);self.assertEqual(self.payload.read_bytes(),original)

if __name__=='__main__':unittest.main()
