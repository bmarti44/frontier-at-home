import concurrent.futures,hashlib,json,pathlib,subprocess,time,urllib.error,urllib.request
root=pathlib.Path(__file__).parent
manifest=json.loads((root/'manifest.json').read_text())
genesis=1595431050;period=30
round_number=int((manifest['frozen_at_unix']-genesis)//period)+2
published=genesis+(round_number-1)*period
if published<=manifest['frozen_at_unix']:raise ValueError('beacon does not follow freeze')
while time.time()<published+2:time.sleep(min(2,published+2-time.time()))
def fetch(host):
 url=f'https://{host}/public/{round_number}'
 for attempt in range(12):
  try:
   with urllib.request.urlopen(url,timeout=20) as r:payload=r.read(16385)
   if len(payload)>16384:raise ValueError('oversized beacon')
   (root/(host+'.json')).write_bytes(payload)
   return json.loads(payload)
  except urllib.error.HTTPError as error:
   if error.code!=404 or attempt==11:raise
   time.sleep(2)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:values=list(pool.map(fetch,['api.drand.sh','api2.drand.sh']))
if values[0]!=values[1] or values[0]['round']!=round_number:raise ValueError('beacon relays disagree')
v=values[0]
cmd=['/home/bmarti44/.nvm/versions/node/v22.22.2/bin/node',str(root/'code/scripts/103_verify_drand_receipt_bundle.mjs'),str(v['round']),v['randomness'],v['signature'],v['previous_signature']]
r=subprocess.run(cmd,env={'PATH':'/usr/bin:/bin','HOME':'/nonexistent','LANG':'C.UTF-8'},capture_output=True,text=True,timeout=30)
(root/'beacon-verifier.stdout').write_text(r.stdout);(root/'beacon-verifier.stderr').write_text(r.stderr)
if r.returncode or r.stdout!='DRAND_BLS_RECEIPT_OK\n':raise ValueError('BLS verification failed')
receipt={**v,'publication_unix':published,'frozen_at_unix':manifest['frozen_at_unix'],'verification':r.stdout.strip(),'relays_agree':True,'seed':int(v['randomness'][:16],16),'verifier':{'sha256':hashlib.sha256((root/'code/scripts/103_verify_drand_receipt_bundle.mjs').read_bytes()).hexdigest()}}
(root/'randomness.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'round':round_number,'publication_unix':published,'seed':receipt['seed'],'verification':receipt['verification']}))
