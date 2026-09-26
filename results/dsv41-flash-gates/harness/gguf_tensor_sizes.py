import struct,sys,re,collections,json
path,total=sys.argv[1],int(sys.argv[2])
f=open(path,'rb')
def rd(fmt): return struct.unpack('<'+fmt,f.read(struct.calcsize('<'+fmt)))
def rstr(): (n,)=rd('Q'); return f.read(n).decode('utf8','replace')
SZ={0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'?',10:'Q',11:'q',12:'d'}
def rval(t):
    if t==8: return rstr()
    if t==9:
        (at,n)=rd('IQ')
        return [rval(at) for _ in range(n)] if n<=64 else ('<array %d of type %d>'%(n,at), [rval(at) for _ in range(n)][:0])
    return rd(SZ[t])[0]
magic=f.read(4); (ver,nt,nkv)=rd('IQQ')
kv={}
for _ in range(nkv):
    k=rstr(); (t,)=rd('I'); kv[k]=rval(t)
align=kv.get('general.alignment',32)
tens=[]
for _ in range(nt):
    name=rstr(); (nd,)=rd('I'); dims=rd('Q'*nd); (tt,off)=rd('IQ'); tens.append((name,dims,tt,off))
pos=f.tell(); data=(pos+align-1)//align*align
tens.sort(key=lambda x:x[3])
sizes={}
for i,(n,d,t,o) in enumerate(tens):
    end=tens[i+1][3] if i+1<len(tens) else total-data
    sizes[n]=(end-o,t,d)
print("version",ver,"tensors",nt,"kv",nkv,"data_start",data)
for k,v in kv.items():
    if 'token' in k and isinstance(v,tuple): continue
    if re.search(r'tokenizer\.ggml\.(tokens|scores|token_type|merges)',k): continue
    print(" ",k,"=",v if not isinstance(v,str) or len(v)<120 else v[:120]+'...')
groups=collections.defaultdict(lambda:[0,0,set()])
for n,(s,t,d) in sizes.items():
    g=re.sub(r'blk\.\d+\.','blk.N.',n)
    groups[g][0]+=s; groups[g][1]+=1; groups[g][2].add(t)
print("\nTOP tensor groups (GiB, count, ggml types):")
for g,(s,c,ts) in sorted(groups.items(),key=lambda x:-x[1][0])[:40]:
    print(f"  {s/2**30:9.3f}  {c:4d}  {sorted(ts)}  {g}")
json.dump({n:[s,t,list(d)] for n,(s,t,d) in sizes.items()},open(sys.argv[3],'w'))
