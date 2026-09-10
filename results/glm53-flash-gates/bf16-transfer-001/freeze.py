"""Prepare a fresh, limited HTTP transport diagnostic; no network payload here."""
from pathlib import Path
import hashlib,json,shutil,subprocess,sys,time
R=Path('/home/bmarti44/spark-deepseek-v4-flash');B=Path.home()/'.cache/glm53-flash';O=B/'bf16-transfer-001';S=Path(__file__).parent
sys.path.insert(0,str(R/'scripts/lib'))
from glm53_probe_capture import inference_lock
from glm53_contract import sha256_file
with inference_lock():
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=R,text=True),'clean source required'
    O.mkdir(exist_ok=False);(O/'code/scripts').mkdir(parents=True);(O/'state').mkdir()
    for name in ['probe.py','test_probe.py','freeze.py','run.py','PROTOCOL.md']:shutil.copyfile(S/name,O/'code'/name)
    shutil.copyfile(R/'scripts/103_verify_drand_receipt_bundle.mjs',O/'code/scripts/103_verify_drand_receipt_bundle.mjs')
    shutil.copyfile(B/'context-freeze-003/fetch-randomness.py',O/'fetch-randomness.py')
    lib=O/'code/scripts/lib';lib.mkdir();shutil.copyfile(R/'scripts/lib/glm53_probe_capture.py',lib/'glm53_probe_capture.py');shutil.copyfile(R/'scripts/lib/glm53_host_evidence.py',lib/'glm53_host_evidence.py')
    shutil.copyfile(B/'bf16-local-feasibility-001/streaming-layout/model-api-blobs.json',O/'model-api-blobs.json')
    python=Path('/usr/bin/python3').resolve();node=Path('/home/bmarti44/.nvm/versions/node/v22.22.2/bin/node');paths=[p for p in sorted((O/'code').rglob('*')) if p.is_file()]+[O/'fetch-randomness.py',O/'model-api-blobs.json',python,node]
    manifest={'scope':'HTTP byte transport only; no model/GPU/qualification result','source_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),'probe':{'sha256':sha256_file(O/'code/probe.py')},'node':str(node),'python':str(python),'unit':'glm53-bf16-transfer-001','files':[{'path':str(p),'size_bytes':p.stat().st_size,'sha256':sha256_file(p)} for p in paths],'frozen_at_unix':time.time()}
    (O/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({'output':str(O),'frozen_at_unix':manifest['frozen_at_unix']}),flush=True)
