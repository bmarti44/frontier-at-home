"""Exercise the exact pinned allocator function with 4097 logical bytes on CPU."""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import torch

source = Path('/home/bmarti44/.cache/glm53-flash/build-source-005/vllm/vllm/v1/worker/utils.py')
parsed = ast.parse(source.read_bytes())
selected = [node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name == 'allocate_kv_cache']
assert len(selected) == 1
class Spec:
    pass
namespace = {'torch': torch, 'KVCacheConfig': object, 'KVCacheLayout': object,
             'MLAAttentionSpec': type('MLA', (), {}), 'UniformTypeKVCacheSpecs': type('Uniform', (), {}),
             'create_kv_cache_views': lambda buf, *args, **kwargs: [buf]}
exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), 'exec'), namespace)
config = SimpleNamespace(kv_cache_tensors=[SimpleNamespace(size=4097, layers=['layer'])],
                         kv_cache_groups=[SimpleNamespace(layer_names=['layer'], kv_cache_spec=Spec())], num_blocks=1)
result = namespace['allocate_kv_cache'](config, torch.device('cpu'), None)
view = result['layer']
assert view.untyped_storage().nbytes() == 8192
assert not bool(torch.count_nonzero(view))
print(json.dumps({'verdict': 'PASS', 'qualification': 'tiny_CPU_allocator_rounding_only',
                  'source': str(source), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                  'logical_bytes': 4097, 'actual_storage_bytes': view.untyped_storage().nbytes(),
                  'all_bytes_zero': True, 'model_loaded': False, 'cuda_used': False}))
