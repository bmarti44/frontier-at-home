#!/usr/bin/env python3
"""Add a default-off, startup-resolved BF16 shard geometry diagnostic candidate."""
import argparse
from pathlib import Path


def patch(source):
    path = source / "src/vllm_exl3/exl3.py"
    text = path.read_text()
    changes = (
        ("import os\n", "import os\n\n# Resolve once at import; no diagnostic work in apply/decode.\n"
         "_GLM53_BF16_SHARD_FIX = os.environ.get(\"GLM53_EXL3_BF16_SHARD_FIX\") == \"1\"\n"),
        ('    logger = init_logger("vllm." + __name__)\n',
         '    logger = init_logger("vllm." + __name__)\n'
         '    if _GLM53_BF16_SHARD_FIX:\n'
         '        logger.info("GLM53_EXL3_BF16_SHARD_FIX=1: preserve BF16 shard geometry")\n'),
        ("        output_partition_sizes = [_exl3_pad128(s) for s in true_out_sizes]\n",
         "        output_partition_sizes = [\n"
         "            s if _GLM53_BF16_SHARD_FIX and i in bf16_shards else _exl3_pad128(s)\n"
         "            for i, s in enumerate(true_out_sizes)\n"
         "        ]\n"),
    )
    for old, new in changes:
        if text.count(old) != 1:
            raise ValueError("BF16 geometry source anchor changed")
        text = text.replace(old, new)
    compile(text, str(path), "exec")
    path.write_text(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    patch(args.source)
