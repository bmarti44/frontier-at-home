"""CPU mutation controls for exact implicit provider selection; no execution."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile

base=Path(__file__).resolve().parent.parent
spec=importlib.util.spec_from_file_location('compiler_selection_fixture',base/'test_review.py')
review=importlib.util.module_from_spec(spec);spec.loader.exec_module(review)
with tempfile.TemporaryDirectory() as d:
    root=Path(d);manifest=review.frozen_fixture(root)
    review.probe.verify_frozen(root,manifest)
    mutations={'PATH':'/tmp:/usr/bin:/bin','CUDA_HOME':'/tmp/other-cuda',
        'CC':'/tmp/cc','CXX':'/tmp/cxx','NVCC_CCBIN':'/tmp/ccbin',
        'FLASHINFER_CXX_LAUNCHER':'/tmp/launcher','LD_LIBRARY_PATH':'/tmp/lib',
        'LD_PRELOAD':'/tmp/lib.so','LD_AUDIT':'/tmp/audit.so'}
    results=[]
    for key,value in mutations.items():
        bad=copy.deepcopy(manifest);bad['environment'][key]=value
        try:review.probe.verify_frozen(root,bad)
        except ValueError as error:results.append({'key':key,'rejected':True,'error':str(error)})
        else:raise AssertionError('accepted provider override '+key)
print(json.dumps({'scope':'Synthetic manifest mutations only; no native or external execution',
    'positive_valid_binding':'PASS','rejections':results,
    'source_sha256':hashlib.sha256((base/'probe.py').read_bytes()).hexdigest()},indent=2))
