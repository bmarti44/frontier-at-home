"""CPU-only launch/native dummy-allocation regression; no model imports."""
import ast,json,types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
RUNTIME=Path('/home/bmarti44/.cache/glm53-flash/native-runtime-002/runtime/lib/python3.12/site-packages/vllm')
source=ROOT/'scripts/47_run_glm53_dev.py'
main=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='main')
start=next(i for i,n in enumerate(main.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='arguments' for t in n.targets))
end=next(i for i,n in enumerate(main.body[start:],start) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='state' for t in n.targets))
profile=json.loads((ROOT/'configs/profiles/glm-5.3-flash/cuda-spark-128g-1m.json').read_text())
args=types.SimpleNamespace(port=8015,prefill_batch=128,skip_autotune=True,release_warmup_cache=True,text_only=False,skip_mm_profiling=True)
ns={'profile':profile,'model':Path('/unused-model'),'args':args,'json':json}
exec(compile(ast.Module(body=main.body[start:end],type_ignores=[]),str(source),'exec'),ns)
argv=ns['arguments'];limits=json.loads(argv[argv.index('--limit-mm-per-prompt')+1]);video=limits['video']
assert isinstance(video,dict) and video.get('count')==1 and video.get('num_frames')==16,limits
assert isinstance(limits['image'],dict) and limits['image']['count']==4
assert argv[argv.index('--max-num-seqs')+1]=='4' and argv[argv.index('--max-model-len')+1]=='262144'
assert all(limits[m].get('width')==512 and limits[m].get('height')==512 for m in ('image','video')),limits
path=RUNTIME/'multimodal/processing/dummy_inputs.py'
tree=ast.parse(path.read_text());method=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_get_dummy_videos')
# Capture the native allocator's requested shape instead of allocating video pixels.
shapes=[]
def full(shape,*a,**kw):shapes.append(shape);return shape
native={'np':types.SimpleNamespace(full=full,uint8='uint8'),'logger':types.SimpleNamespace(warning=lambda *a:None)}
code='from __future__ import annotations\n'+ast.unparse(method)
exec(compile(code,str(path),'exec'),native)
options=types.SimpleNamespace(num_frames=video['num_frames'],width=video['width'],height=video['height'])
for computed in (8,16,1200):
 result=native['_get_dummy_videos'](None,width=4096,height=2048,num_frames=computed,num_videos=1,overrides=options)
 assert result==[(min(computed,16),512,512,3)],result
print(json.dumps({'scope':'CPU native allocation-shape and launch regression','verdict':'PASS','frames':[s[0] for s in shapes],'pixel_arrays_allocated':0}))
