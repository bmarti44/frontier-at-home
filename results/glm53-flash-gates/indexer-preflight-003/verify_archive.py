import hashlib, importlib.util, json, pathlib, tarfile, tempfile, time
root=pathlib.Path(__file__).resolve().parent
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
restore=module('restore',root/'restore_artifacts.py')
started=time.time()
with tempfile.TemporaryDirectory(prefix='glm53-archive003-') as temporary:
 destination=pathlib.Path(temporary)/'restored'
 result=restore.restore(root,destination)
 manifest=json.loads((root/'archive.json').read_text())
 original=pathlib.Path(manifest['local_attempt'])
 for name in manifest['copied_files']|manifest['compressed_files']:
  assert hashlib.sha256((original/name).read_bytes()).digest()==hashlib.sha256((destination/name).read_bytes()).digest(),name
 with tarfile.open(destination/'state.tar.gz') as bundle:
  inventory=manifest['bundles']['state']['file_inventory']
  files={m.name:m for m in bundle.getmembers() if m.isfile()}
  print('tar sample',list(files)[:2],flush=True)
  assert len(files)==len(inventory)
  for name,record in inventory.items():
   data=bundle.extractfile(files['state/'+name]).read()
   assert len(data)==record['size_bytes'] and hashlib.sha256(data).hexdigest()==record['sha256']
 scorer=module('frozen_indexer',destination/'code/scripts/45_probe_glm53_indexer.py')
 rows=[json.loads(line) for line in (destination/'checks/raw.jsonl').read_text().splitlines()]
 scored=scorer.score_capture(destination/'checks',rows,7631521019026542407)
 assert scored==json.loads((destination/'checks/summary.json').read_text())['checks']
 try: restore.restore(root,destination)
 except FileExistsError: pass
 else: raise AssertionError('existing destination accepted')
 for invalid in ('../escape','/absolute','a/../escape','a//b','./a','.'):
  try: restore.relative(invalid)
  except ValueError: pass
  else: raise AssertionError(invalid)
 bad=pathlib.Path(temporary)/'bad'; bad.mkdir(); (bad/'x').write_bytes(b'changed')
 try: restore.verify(bad/'x',{'size_bytes':7,'sha256':hashlib.sha256(b'correct').hexdigest()})
 except ValueError: pass
 else: raise AssertionError('wrong digest accepted')
 (bad/'link').symlink_to(bad/'x')
 try: restore.regular(bad,'link')
 except ValueError: pass
 else: raise AssertionError('symlink accepted')
 result.update(original_files_verified=len(manifest['copied_files'])+len(manifest['compressed_files']),state_files_verified=len(files),frozen_scorer_cases=len(scored),valid_logits=sum(x['valid_logit_elements'] for case in scored for x in case['logits']),controls=['existing destination rejected','noncanonical paths rejected','wrong digest rejected','source symlink rejected'],elapsed_seconds=time.time()-started)
 (root/'archive-verification.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result),flush=True)
