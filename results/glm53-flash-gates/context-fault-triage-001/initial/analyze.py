"""CPU-only address arithmetic; never imports or executes the model runtime."""
import ast
import hashlib
import json
from pathlib import Path
import re
import time

OUT = Path(__file__).resolve().parent
BASE = Path('/home/bmarti44/.cache/glm53-flash')
VLLM = BASE / 'native-runtime-002/runtime/lib/python3.12/site-packages/vllm'
LOG = Path('/home/bmarti44/.local/state/glm52-crashlog/20260909-171537-glm53-context-1370392/cmd.log')

def require(condition, message):
    if not condition:
        raise ValueError(message)

sources = [LOG, BASE / 'model-weights-001/config.json',
           VLLM / 'models/glm5next/nvidia/kda.py',
           VLLM / 'model_executor/layers/mamba/ops/causal_conv1d.py',
           VLLM / 'v1/attention/backends/gdn_attn.py',
           VLLM / 'v1/attention/backends/utils.py']
with LOG.open() as stream:
    dumps = [(n, line) for n, line in enumerate(stream, 1)
             if 'Dumping scheduler output for model execution:' in line]
require(len(dumps) == 1, 'expected one failed scheduler dump')
line_number, dump = dumps[0]
computed = ast.literal_eval(re.search(r'num_computed_tokens=(\[[^]]*\])', dump)[1])
outputs = ast.literal_eval(re.search(r'num_output_tokens=(\[[^]]*\])', dump)[1])
scheduled = re.search(r'num_scheduled_tokens=\{([^}]*)\}', dump)[1]
chunks = [int(v) for v in re.findall(r': (\d+)', scheduled)]
require(computed == [52992, 52928, 52896, 52864], 'unexpected failure geometry')
require(outputs == [0] * 4 and chunks == [32] * 4, 'not the pure-prefill batch')

config = json.loads(sources[1].read_text())
config = config.get('text_config', config)['linear_attn_config']
dim = 3 * config['num_heads'] * config['head_dim']
width = config['short_conv_kernel_size']
require((dim, width) == (24576, 4), 'unexpected convolution geometry')
tree = ast.parse(sources[2].read_text())
calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
         and isinstance(n.func, ast.Name) and n.func.id == 'causal_conv1d_fn']
require(len(calls) == 1, 'ambiguous convolution call')
keywords = {k.arg for k in calls[0].keywords}
require(not keywords.intersection({'block_idx_first_scheduled_token',
    'block_idx_last_scheduled_token', 'initial_state_idx', 'num_computed_tokens'}),
    'APC/history offsets explicitly supplied; analysis no longer applies')

# Model the rebased four-request convolution launch, not GPU tensors. Prefix
# history is absent from these equations. Real state IDs were not captured.
rows = []
for sequence in range(4):
    for chunk in range(4):
        begin = sequence * 32 + chunk * 8
        end = begin + 8
        require(sequence * 32 <= begin < end <= (sequence + 1) * 32,
                'convolution chunk crosses its request')
        # Channel-last input after transpose, and SD state after transpose.
        rows.append({'scope': 'CPU arithmetic, not observed device addresses',
                     'sequence': sequence, 'chunk': chunk,
                     'input_element_min': begin * dim,
                     'input_element_max': end * dim - 1,
                     'state_row_element_max': (width - 1) * dim - 1})
require(max(r['input_element_max'] for r in rows) == 128 * dim - 1,
        'input bound mismatch')
(OUT / 'raw.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
summary = {
    'scope': 'bounded CPU geometry falsifier; no model/kernel execution',
    'verdict': 'NO_RESULT',
    'reason': 'no originating CUDA kernel or real state/cache index values captured',
    'arithmetic_checks_passed': True,
    'failed_scheduler_line': line_number,
    'computed_tokens': computed, 'scheduled_tokens': chunks,
    'rebased_query_start_loc': [0, 32, 64, 96, 128],
    'convolution_programs': len(rows), 'merged_channels': dim,
    'formula': 'input offset = rebased token * 24576 + channel; SD state row offset = history tap * 24576 + channel',
    'finding': 'Absolute 529xx history does not itself enlarge these local convolution offsets. Invalid state IDs, corrupt metadata, data-dependent failures and earlier kernels remain untested.',
    'time_unix': time.time(),
}
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
files = []
for path in [*sources, Path(__file__), OUT / 'raw.jsonl', OUT / 'summary.json']:
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    files.append({'path': str(path), 'sha256': digest, 'size_bytes': path.stat().st_size})
(OUT / 'manifest.json').write_text(json.dumps({'scope': summary['scope'], 'files': files}, indent=2) + '\n')
print(json.dumps(summary, indent=2))
