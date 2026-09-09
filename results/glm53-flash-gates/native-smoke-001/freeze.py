import hashlib,json,pathlib,shutil,subprocess,sys,time
root=pathlib.Path('/home/bmarti44/spark-deepseek-v4-flash')
base=pathlib.Path('/home/bmarti44/.cache/glm53-flash')
output=base/'native-smoke-001';output.mkdir()
code=output/'code'
files=['scripts/35_smoke_glm53_native.py','scripts/lib/glm53_contract.py','configs/build-manifests/glm53-flash-sources.json','scripts/103_verify_drand_receipt_bundle.mjs']
for name in files:
 target=code/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(root/name,target)
sys.path.insert(0,str(code/'scripts/lib'))
from glm53_contract import sha256_file,strict_json,verify_inventory
runtime=base/'native-runtime-001/runtime';inventory=base/'native-runtime-001/inventory.json'
identities=verify_inventory(runtime,strict_json(inventory))
if subprocess.check_output(['git','-C',str(root),'status','--porcelain'],text=True):raise ValueError('repository is not clean')
prepared=base/'build-source-004'
state=output/'state';state.mkdir()
env={'HOME':str(state),'PATH':f'{runtime}/bin:/usr/local/cuda-13.0/bin:/usr/bin:/bin','LANG':'C.UTF-8','CUDA_HOME':'/usr/local/cuda-13.0','CUDA_VISIBLE_DEVICES':'0','TORCH_COMPILE_DISABLE':'1','PYTHONDONTWRITEBYTECODE':'1','NVIDIA_TF32_OVERRIDE':'0','MAX_JOBS':'2','NVCC_THREADS':'1','TRITON_CACHE_DIR':str(state/'triton'),'TORCH_EXTENSIONS_DIR':str(state/'torch-extensions'),'CUDA_CACHE_PATH':str(state/'cuda-cache')}
manifest={'schema_version':1,'qualification':'synthetic_native_smoke_only','source_revision':subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),'runtime':{'root':str(runtime),'manifest':str(inventory),'sha256':sha256_file(inventory),'files':len(identities)},'code':{name:{'sha256':sha256_file(code/name)} for name in files},'prepared':{'root':str(prepared),'sha256':sha256_file(prepared/'manifest.json')},'test_files':{name:{'sha256':sha256_file(prepared/'vllm-exl3/tests'/name)} for name in ('test_exl3_linear.py','test_native_moe_contract.py')},'tools':{str(p):{'sha256':sha256_file(p)} for p in (runtime/'bin/python3',pathlib.Path('/usr/bin/git'),pathlib.Path('/home/bmarti44/.nvm/versions/node/v22.22.2/bin/node'),root/'results/glm52-gates/harness/glm_cgroup_run.sh',root/'results/glm52-gates/harness/glm_safe_run.sh')},'environment':env,'safety':{'minimum_start_gib':110,'kill_floor_gib':40,'timeout_seconds':600},'argv_without_seed':[str(runtime/'bin/python3'),'-I','-B',str(code/'scripts/35_smoke_glm53_native.py'),'--prepared',str(prepared),'--output',str(output/'checks')],'seed_rule':'uint64 from first16 hex characters of BLS-verified post-freeze drand randomness','frozen_at_unix':time.time(),'model_weights':'none; synthetic seeded tensors only'}
(output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
shutil.copyfile(__file__,output/'freeze.py')
print(json.dumps({'freeze':str(output/'manifest.json'),'frozen_at_unix':manifest['frozen_at_unix'],'files_verified':len(identities)}))
