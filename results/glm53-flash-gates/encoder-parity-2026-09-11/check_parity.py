#!/usr/bin/env python3
"""Live parity check: encoding_glm53 text vs the served chat template.

For each case and effort: (1) the encoder's text and the on-disk Jinja
template tokenize to identical ids with the served tokenizer.json;
(2) /v1/completions on the encoder text and /v1/chat/completions on the
messages report the same prompt_tokens; (3) the top-5 first-token logprobs
agree in ranking (values may differ by kernel-level nondeterminism).
"""
import json, urllib.request, importlib.util, sys
from tokenizers import Tokenizer
import jinja2
ROOT = "/home/bmarti44/spark-deepseek-v4-flash"
spec = importlib.util.spec_from_file_location("enc", ROOT + "/vendor/official-encoding/encoding/encoding_glm53.py")
enc = importlib.util.module_from_spec(spec); spec.loader.exec_module(enc)
M = "/home/bmarti44/models/glm-5.3-flash/k2-densek4-mtp"
tok = Tokenizer.from_file(M + "/tokenizer.json")
tpl = jinja2.Environment(extensions=["jinja2.ext.loopcontrols"]).from_string(open(M + "/chat_template.jinja").read())
BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8015"
def post(path, body):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=120))
cases = [
  [{"role":"user","content":"Janet has 3 apples. She buys 5 more. How many apples?"}],
  [{"role":"system","content":"Answer with a single letter."},{"role":"user","content":"Q: 2+2?\nA. 3\nB. 4\nAnswer:"}],
  [{"role":"user","content":"hi"},{"role":"assistant","content":"hello","reasoning_content":"greet"},{"role":"user","content":"def f(x):\n    return x*2  # ünïcödé 数学"}],
]
rows, bad = [], 0
for effort in ("low", "max"):
  for i, msgs in enumerate(cases):
    text = enc.encode_messages(msgs, thinking_mode="thinking", reasoning_effort=effort)
    off = tpl.render(messages=msgs, add_generation_prompt=True, clear_thinking=True, reasoning_effort=effort)
    ids_mine = tok.encode(text, add_special_tokens=False).ids
    ids_off = tok.encode(off, add_special_tokens=False).ids
    c = post("/v1/completions", {"model":"glm-5.3-flash","prompt":text,"max_tokens":1,"temperature":0,"logprobs":5,"seed":1})
    ch = post("/v1/chat/completions", {"model":"glm-5.3-flash","messages":msgs,"max_tokens":1,"temperature":0,"logprobs":True,"top_logprobs":5,"seed":1,
                                        "chat_template_kwargs":{"reasoning_effort":effort,"clear_thinking":True}})
    pt_c, pt_ch = c["usage"]["prompt_tokens"], ch["usage"]["prompt_tokens"]
    top_c = c["choices"][0]["logprobs"]["top_logprobs"][0]
    top_ch = {d["token"]: d["logprob"] for d in ch["choices"][0]["logprobs"]["content"][0]["top_logprobs"]}
    rank_c = sorted(top_c, key=top_c.get, reverse=True); rank_ch = sorted(top_ch, key=top_ch.get, reverse=True)
    max_abs = max(abs(top_c[k] - top_ch[k]) for k in top_c) if set(top_c) == set(top_ch) else None
    row = {"case": i, "effort": effort, "ids_equal_to_template": ids_mine == ids_off, "n_ids": len(ids_mine),
           "server_prompt_tokens_completions": pt_c, "server_prompt_tokens_chat": pt_ch,
           "top5_rank_equal": rank_c == rank_ch, "top5_max_abs_logprob_delta": max_abs, "top5_completions": top_c, "top5_chat": top_ch}
    ok = row["ids_equal_to_template"] and pt_c == pt_ch == len(ids_mine) and row["top5_rank_equal"]
    row["pass"] = ok; bad += not ok; rows.append(row)
    print(json.dumps({k: v for k, v in row.items() if not k.startswith("top5_c")}))
print("MISMATCHES:", bad)
json.dump({"base_url": BASE, "rows": rows, "mismatches": bad}, open(sys.argv[2], "w") if len(sys.argv) > 2 else sys.stdout, indent=1)
sys.exit(1 if bad else 0)
