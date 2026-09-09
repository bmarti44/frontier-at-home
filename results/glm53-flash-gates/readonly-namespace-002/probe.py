import errno,json,pathlib
p=pathlib.Path("/home/bmarti44/.cache/glm53-flash/readonly-namespace-002/sentinel")
try:
 p.write_text("changed\n")
except OSError as e:
 print(json.dumps({"errno":e.errno,"readonly":e.errno==errno.EROFS}))
 raise SystemExit(0 if e.errno==errno.EROFS else 2)
raise SystemExit(3)
