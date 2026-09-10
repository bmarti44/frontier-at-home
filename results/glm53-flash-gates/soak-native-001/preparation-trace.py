import sys,importlib.util,tempfile,json
from pathlib import Path
path=Path('/home/bmarti44/spark-deepseek-v4-flash/results/glm53-flash-gates/soak-native-001/run.py')
spec=importlib.util.spec_from_file_location('target',path);api=importlib.util.module_from_spec(spec);spec.loader.exec_module(api)
def trace(frame,event,arg):
 if frame.f_code.co_filename==str(path) and frame.f_code.co_name=='prepare' and event=='line' and frame.f_lineno==82:
  d=frame.f_locals;print(json.dumps({'worker':d['worker'],'target':d['target'],'rendered_tokens':len(d['ids'])}),flush=True)
 return trace
sys.settrace(trace)
with tempfile.TemporaryDirectory(prefix='glm53-soak-trace-') as root:api.prepare(Path(root)/'inputs','0'*64)
