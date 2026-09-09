import argparse,hashlib,json,pathlib,struct
from wheel.wheelfile import WheelFile
p=argparse.ArgumentParser();p.add_argument('wheel',type=pathlib.Path);p.add_argument('--required',action='append',required=True);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
native=[]
with WheelFile(a.wheel) as wheel:
 names=wheel.namelist()
 if len(names)!=len(set(names)):raise ValueError('duplicate wheel member')
 for required in a.required:
  if required not in names:raise ValueError('missing required native member: '+required)
 for name in names:
  if name.endswith('/'):continue
  b=wheel.read(name)
  if name.endswith('.so'):
   if b[:4]!=b'\x7fELF' or b[4:6]!=b'\x02\x01' or struct.unpack_from('<H',b,18)[0]!=183:
    raise ValueError('native member is not AArch64 ELF64: '+name)
   native.append({'path':name,'size_bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()})
receipt={'schema_version':1,'qualification':'wheel_RECORD_and_native_layout_only','verdict':'PASS','wheel':{'path':str(a.wheel),'sha256':hashlib.sha256(a.wheel.read_bytes()).hexdigest(),'size_bytes':a.wheel.stat().st_size},'members':len(names),'native':native,'required':a.required,'scorer':{'path':str(pathlib.Path(__file__)),'sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}}
a.output.write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'verdict':'PASS','members':len(names),'native_members':len(native)}))
