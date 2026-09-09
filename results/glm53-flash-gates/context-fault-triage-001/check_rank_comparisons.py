"""Small CPU comparison check; does not execute CUDA or reproduce the crash."""
import hashlib
import json
from pathlib import Path

source = Path('/home/bmarti44/.cache/glm53-flash/build-source-005/vllm/csrc/libtorch_stable/sampler.cu')
text = source.read_text()
comparison = 'if (logit < otherLogit || (logit == otherLogit && i < j))'
if text.count(comparison) != 1:
    raise ValueError('source comparison changed')

def ranks(values):
    return [sum(value < other or (value == other and i < j)
                for j, other in enumerate(values))
            for i, value in enumerate(values)]

rows = []
for label, values in [('finite_with_ties', [3.0, 1.0, 3.0, -2.0]),
                      ('all_nonfinite_nan', [float('nan')] * 4)]:
    indices = ranks(values)
    rows.append({'case': label, 'comparison_ranks': indices,
                 'permutation_of_four_slots': sorted(indices) == list(range(4))})
if not rows[0]['permutation_of_four_slots'] or rows[1]['permutation_of_four_slots']:
    raise ValueError('unexpected comparison result')
result = {'scope': 'CPU insertion-final-phase comparison only',
          'verdict': 'NO_RESULT',
          'reason': 'No proof of nonfinite native logits, histogram admission to this phase, or downstream CUDA fault',
          'source': {'path': str(source), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest()},
          'cases': rows,
          'finding': 'If four NaNs reach this comparison phase, all receive rank zero. This motivates a bounded native top-k test, not a production patch.'}
Path(__file__).with_name('rank-comparison-result.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
