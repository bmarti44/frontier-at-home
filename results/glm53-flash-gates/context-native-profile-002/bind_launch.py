"""Bind actual launch observations without changing the prepared request inputs."""
import importlib.util,json,shutil,sys
from pathlib import Path
ROOT=Path('/home/bmarti44/spark-deepseek-v4-flash')
spec=importlib.util.spec_from_file_location('probe',ROOT/'scripts/48_probe_glm53_context.py');probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)
out,server=map(Path,sys.argv[1:]);manifest=probe.verify(out)
planned=probe.read(out/'server-launch.json');actual=probe.launch_check(server)
if planned.get('scope')!='declared fixture configuration only; no running server observation':raise ValueError('expected preregistered fixture configuration')
if planned['arguments']!=actual['arguments']:raise ValueError('actual profile arguments differ from frozen fixture configuration')
for name in ('prelaunch-manifest.json','planned-server-launch.json'):
 if (out/name).exists():raise ValueError('launch binding already exists')
shutil.copyfile(out/'manifest.json',out/'prelaunch-manifest.json')
shutil.copyfile(out/'server-launch.json',out/'planned-server-launch.json')
shutil.copyfile(server/'launch.json',out/'server-launch.json')
manifest['files']['server-launch.json']=probe.sha(out/'server-launch.json')
manifest['launch_binding']={'prelaunch_manifest':{'sha256':probe.sha(out/'prelaunch-manifest.json')},'planned_configuration':{'sha256':probe.sha(out/'planned-server-launch.json')},'binding_program':{'path':str(Path(__file__).resolve()),'sha256':probe.sha(Path(__file__))}}
probe.write(out/'manifest.json',manifest);probe.verify(out)
print(json.dumps({'actual_launch_bound':True,'requests_and_fixtures_unchanged':True}),flush=True)
