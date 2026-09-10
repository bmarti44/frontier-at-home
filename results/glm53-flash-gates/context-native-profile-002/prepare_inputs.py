"""Prepare fresh direct-context fixtures with the preregistered output allowance.

Uses the unchanged repository fixture builder and scorer. Only max_tokens is
changed, before final arm-input hashes are recorded; prompt token IDs stay exact.
Run this frozen preparation program after obtaining the public seed.
"""
import hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path('/home/bmarti44/spark-deepseek-v4-flash')
PYTHON='/home/bmarti44/.cache/glm53-flash/native-runtime-002/runtime/bin/python3'
out,seed,server=map(str,sys.argv[1:])
subprocess.run([PYTHON,'-I','-B',str(ROOT/'scripts/48_probe_glm53_context.py'),'prepare','--output',out,'--seed',seed,'--server',server],check=True)
out=Path(out);manifest=json.loads((out/'manifest.json').read_text())
for slot in range(4):
 path=out/f'{slot}-request.json';body=json.loads(path.read_text())
 if body['max_tokens']!=256:raise ValueError('unexpected inherited output configuration')
 body['max_tokens']=2048
 path.write_text(json.dumps(body,indent=2,allow_nan=False)+'\n')
 manifest['files'][path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
manifest['request_configuration']={'max_tokens':2048,'change':'output allowance only; prompt construction and scorer unchanged','preparation_source':{'path':str(Path(__file__).resolve()),'sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
