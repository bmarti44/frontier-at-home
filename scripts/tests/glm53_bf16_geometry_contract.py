#!/usr/bin/env python3
"""Real CPU weight-constructor checks; no model weights or CUDA initialization."""
import argparse
from pathlib import Path
import sys
import unittest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source.resolve() / "src"))
    import torch
    import vllm.distributed as dist
    from vllm_exl3.exl3 import Exl3Config, Exl3LinearMethod
    dist.get_tensor_model_parallel_rank = lambda: 0
    dist.get_tensor_model_parallel_world_size = lambda: 1
    prefix = "model.language_model.layers.0.self_attn.in_proj_qkvbfg_a"

    class GeometryContract(unittest.TestCase):
        def construct(self, shards, bf16, inputs=4096):
            cfg = Exl3Config(bits=4, non_routed_exl3={"codebook": "mul1", "layers": {
                prefix: {"bits": 4, "bf16_shards": bf16}}})
            method = Exl3LinearMethod(cfg, bits=4)
            layer = torch.nn.Module()
            layer.prefix = prefix
            layer.quant_method = method
            layer.weight_loader = lambda *args, **kwargs: None
            method.create_weights(layer, inputs, shards, inputs, sum(shards), torch.bfloat16)
            return layer

        def test_mixed_bf16_shards_keep_exact_checkpoint_geometry(self):
            for width in (512, 8192):
                with self.subTest(exl3_width=width):
                    shards = [width, width, width, 64, 128, 128]
                    layer = self.construct(shards, [3, 4, 5])
                    self.assertEqual(layer._exl3_linear_output_partition_sizes, shards)
                    self.assertFalse(layer._exl3_linear_padded)
                    self.assertEqual(tuple(layer.weight.shape), (320, 4096))
                    offset = 0
                    for shard, rows in ((3, 64), (4, 128), (5, 128)):
                        values = torch.full((rows, 4096), shard, dtype=torch.bfloat16)
                        layer.weight.weight_loader(layer.weight, values, shard)
                        self.assertTrue(torch.equal(layer.weight[offset:offset + rows], values))
                        offset += rows

        def test_unaligned_quantized_fused_shard_still_rejects(self):
            with self.assertRaisesRegex(NotImplementedError, "padded linear geometry"):
                self.construct([512, 512, 64, 64, 128, 128], [3, 4, 5])

        def test_aligned_quantized_layer_is_unchanged(self):
            layer = self.construct([512, 512, 1024], [])
            self.assertEqual(layer._exl3_linear_output_partition_sizes, [512, 512, 1024])
            self.assertFalse(layer._exl3_linear_padded)

        def test_mixed_unaligned_input_still_rejects(self):
            with self.assertRaisesRegex(NotImplementedError, "padded linear geometry"):
                self.construct([512, 512, 512, 64, 128, 128], [3, 4, 5], inputs=4000)

        def test_standalone_padded_exl3_is_unchanged(self):
            layer = self.construct([64], [], inputs=4000)
            self.assertEqual(layer._exl3_linear_output_partition_sizes, [128])
            self.assertEqual(layer._exl3_linear_input_size_per_partition, 4096)
            self.assertTrue(layer._exl3_linear_padded)

        def test_cuda_was_not_initialized(self):
            self.assertFalse(torch.cuda.is_initialized())

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(GeometryContract))
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU constructor probe unexpectedly initialized CUDA")
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
