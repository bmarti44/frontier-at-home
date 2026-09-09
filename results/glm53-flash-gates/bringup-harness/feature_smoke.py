import base64,concurrent.futures,json,struct,sys,time,urllib.request,zlib
from pathlib import Path
root=Path(sys.argv[1]);key=(root/'api-key').read_text().strip()
def call(name,body):
 body={'model':'glm-5.3-flash','temperature':0,'max_tokens':512,**body}
 (root/(name+'-request.json')).write_text(json.dumps(body)+'\n')
 req=urllib.request.Request('http://127.0.0.1:8015/v1/chat/completions',data=json.dumps(body).encode(),headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
 start=time.time()
 try:
  with urllib.request.urlopen(req,timeout=300) as response: data=json.load(response)
  (root/(name+'-response.json')).write_text(json.dumps({'start':start,'end':time.time(),'response':data},indent=2)+'\n')
  return data['choices'][0]
 except Exception as error:
  (root/(name+'-error.json')).write_text(json.dumps({'error':str(error),'time':time.time(),'body':error.read().decode() if hasattr(error,'read') else None})+'\n');raise
mode=sys.argv[2]
if mode=='slots':
 def slot(n):
  r=call('slot-'+str(n),{'messages':[{'role':'user','content':f'What is {n} + {n}? Answer in one short sentence.'}]})
  assert r['finish_reason']=='stop' and str(2*n) in (r['message'].get('content') or ''),r
  return {'slot':n,'choice':r}
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool: result=list(pool.map(slot,range(2,6)))
elif mode=='tools':
 result=call('tool-smoke',{'messages':[{'role':'user','content':'Use the get_weather tool to check the weather in Paris.'}],'tools':[{'type':'function','function':{'name':'get_weather','description':'Get weather for a city','parameters':{'type':'object','properties':{'city':{'type':'string'}},'required':['city']}}}],'tool_choice':'auto'})
 assert result['message'].get('tool_calls'),result
elif mode in ('image','images4'):
 def chunk(kind,data):return struct.pack('>I',len(data))+kind+data+struct.pack('>I',zlib.crc32(kind+data)&0xffffffff)
 def square(rgb):
  return b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',224,224,8,2,0,0,0))+chunk(b'IDAT',zlib.compress((b'\0'+bytes(rgb)*224)*224))+chunk(b'IEND',b'')
 colors=[('red',(255,0,0))] if mode=='image' else [('red',(255,0,0)),('green',(0,255,0)),('blue',(0,0,255)),('yellow',(255,255,0))]
 content=[{'type':'text','text':'Name the dominant color of each image in the order shown. Answer with only the color names in order.'}]
 for name,rgb in colors:
  png=square(rgb); (root/(name+'-square.png')).write_bytes(png)
  content.append({'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(png).decode()}})
 result=call(mode+'-smoke',{'messages':[{'role':'user','content':content}]})
 answer=(result['message'].get('content') or '').lower()
 assert result['finish_reason']=='stop',result
 offset=0
 for name,_ in colors:
  position=answer.find(name,offset); assert position>=0,result
  offset=position+len(name)
elif mode=='video':
 video=Path('/tmp/glm53-blue-16frames.mp4').read_bytes()
 (root/'blue-16frames.mp4').write_bytes(video)
 result=call('video-smoke',{'messages':[{'role':'user','content':[{'type':'text','text':'What color fills this video? Answer in one short sentence.'},{'type':'video_url','video_url':{'url':'data:video/mp4;base64,'+base64.b64encode(video).decode()}}]}]})
 assert 'blue' in (result['message'].get('content') or '').lower(),result
else:raise ValueError(mode)
print(json.dumps(result),flush=True)
