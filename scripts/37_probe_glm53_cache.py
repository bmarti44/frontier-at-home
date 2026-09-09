#!/usr/bin/env python3
"""Allocate real GLM cache storage and reserve four slots, without model weights.

Use only inside the existing inference-lock/cgroup wrapper, after freezing the
runtime, metadata and this scorer. Token IDs express reservations, not evaluated
inputs. This cannot establish context capability, fidelity or serving speed.
"""
import argparse
import json
from pathlib import Path
import random
import sys
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/lib"))
from glm53_contract import sha256_file, strict_json


def require(condition, message):
    if not condition:
        raise ValueError(message)


def reserve_four(manager, requests, seed, record):
    require(len(requests) == 5, "five distinct request fixtures required")
    identity = lambda request: getattr(request, "request_id", request)
    require(len({identity(r) for r in requests}) == 5, "distinct request IDs required")
    order = list(requests)
    random.Random(seed).shuffle(order)
    initial = manager.block_pool.get_num_free_blocks()
    require(initial == 144, "unexpected initial pool capacity")
    snapshots = {}
    all_ids = set()
    for index, request in enumerate(order[:4]):
        blocks = manager.allocate_slots(request, num_new_tokens=262144, full_sequence_must_fit=True)
        require(blocks is not None, "required full request reservation rejected")
        groups = blocks.get_block_ids()
        require([len(group) for group in groups] == [31, 1, 1, 1, 1, 1], "unexpected reservation group coverage")
        ids = [block for group in groups for block in group]
        require(all(type(block) is int and block > 0 for block in ids), "null or invalid physical block")
        require(len(set(ids)) == 36 and not all_ids.intersection(ids), "reservation blocks must be distinct")
        all_ids.update(ids)
        snapshots[identity(request)] = [list(group) for group in groups]
        free = manager.block_pool.get_num_free_blocks()
        require(free == 144 - 36 * (index + 1), "incorrect live pool accounting")
        record({"event": "reserve", "request_id": identity(request), "block_ids": groups, "free_blocks": free})
    fifth = manager.allocate_slots(order[4], num_new_tokens=262144, full_sequence_must_fit=True)
    require(fifth is None, "fifth full request must reject")
    require(manager.block_pool.get_num_free_blocks() == 0, "fifth rejection changed pool capacity")
    for request in order[:4]:
        current = manager.get_blocks(identity(request)).get_block_ids()
        require([list(group) for group in current] == snapshots[identity(request)], "live reservation changed")
    record({"event": "fifth_rejected", "live_requests": 4, "distinct_blocks": len(all_ids)})
    for request in reversed(order[:4]):
        manager.free(request)
    require(manager.block_pool.get_num_free_blocks() == initial, "freeing all requests must restore pool capacity")
    record({"event": "restored", "free_blocks": initial})


def run_native(metadata, expected, seed, record):
    import torch
    from vllm import SamplingParams
    from vllm.config import set_current_vllm_config
    from vllm.engine.arg_utils import EngineArgs
    from vllm.platforms import current_platform
    from vllm.model_executor.layers.attention.mla_attention import MLAAttention, _canonicalize_sparse_mla_kv_cache_dtype
    from vllm.model_executor.layers.mamba.abstract import MambaBase
    from vllm.models.glm5next.nvidia.model import Glm5NextForCausalLM
    from vllm.models.glm5next.nvidia.attention import Glm5NextIndexerCache, Glm5NextTailCache
    from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import FlashInferMLASparseSM120Backend
    from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerBackend, KpoolTailBackend
    from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
    from vllm.v1.attention.selector import get_mamba_attn_backend
    from vllm.v1.attention.backends.utils import get_supported_kv_cache_layouts, resolve_kv_cache_layout
    from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
    from vllm.v1.core.kv_cache_utils import get_kv_cache_groups, get_kv_cache_config_from_groups, generate_scheduler_kv_cache_config, resolve_kv_cache_block_sizes
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.kv_cache_interface import MambaSpec
    from vllm.v1.request import Request
    from vllm.v1.worker.utils import allocate_kv_cache

    require(torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 1), "SM121 required")
    cfg = EngineArgs(model=str(metadata), skip_tokenizer_init=True, dtype="bfloat16", quantization="exl3",
                     max_model_len=262144, max_num_seqs=4, max_num_batched_tokens=2048,
                     enable_chunked_prefill=True, enable_prefix_caching=False, kv_cache_dtype="fp8",
                     compilation_config={"mode": 0}).create_engine_config()
    require(cfg.speculative_config is None and not cfg.cache_config.enable_prefix_caching, "unexpected cache/speculation mode")
    require(cfg.cache_config.num_gpu_blocks_override is None, "block override forbidden")
    require(all(getattr(cfg.parallel_config, name) == 1 for name in (
        "tensor_parallel_size", "pipeline_parallel_size", "decode_context_parallel_size", "prefill_context_parallel_size")), "single-device baseline required")
    backend = FlashInferMLASparseSM120Backend
    with set_current_vllm_config(cfg):
        cfg.cache_config.cache_dtype = _canonicalize_sparse_mla_kv_cache_dtype(backend, cfg.cache_config.cache_dtype)
        cfg.cache_config.block_size = backend.get_preferred_block_size(cfg.cache_config.block_size)
        current_platform._align_hybrid_block_size(cfg, backend)
        require(cfg.cache_config.block_size == expected["manager_block_tokens"], "normalized attention block mismatch")
        require(cfg.cache_config.mamba_block_size == 262144 and cfg.cache_config.mamba_cache_mode == "none", "normalized Mamba mode mismatch")
        shapes = Glm5NextForCausalLM.get_mamba_state_shape_from_config(cfg)
        dtypes = Glm5NextForCausalLM.get_mamba_state_dtype_from_config(cfg)
        mamba = SimpleNamespace(get_state_shape=lambda: shapes, get_state_dtype=lambda: dtypes,
                                mamba_type=MambaAttentionBackendEnum.GDN_ATTN)
        mla = SimpleNamespace(kv_cache_dtype=cfg.cache_config.cache_dtype, head_size=512,
                              sliding_window=None, non_causal_multi_token_decode=False)
        specs = {}
        layers = cfg.model_config.hf_text_config.layer_types
        require(layers.count("linear_attention") == expected["kda_layers"] and
                layers.count("deepseek_sparse_attention") == expected["mla_layers"] and len(layers) == 45, "model layer coverage mismatch")
        for index, kind in enumerate(layers):
            prefix = f"model.language_model.layers.{index}.self_attn"
            if kind == "linear_attention":
                specs[prefix] = MambaBase.get_kv_cache_spec(mamba, cfg)
            else:
                specs[prefix + ".attn"] = MLAAttention.get_kv_cache_spec(mla, cfg)
                for suffix, cls, head, dtype in ((".indexer.k_cache", Glm5NextIndexerCache, 132, torch.uint8),
                                                 (".indexer.tail_cache", Glm5NextTailCache, 128, torch.bfloat16)):
                    cache = cls(head_dim=head, dtype=dtype, prefix=prefix + suffix,
                                cache_config=cfg.cache_config, index_kpool=4)
                    specs[prefix + suffix] = cache.get_kv_cache_spec(cfg)
        register_all_kvcache_specs(cfg)
        groups = get_kv_cache_groups(cfg, specs)
        require(len(groups) == expected["groups"], "group coverage mismatch")
        require([len(g.layer_names) for g in groups if isinstance(g.kv_cache_spec, MambaSpec)] == expected["kda_group_layer_counts"], "KDA group coverage mismatch")
        worker = get_kv_cache_config_from_groups(cfg, groups, expected["unique_backing_bytes"])
        require(worker.num_blocks == expected["shared_pool_blocks"], "shared block count mismatch")
        supported = get_supported_kv_cache_layouts([backend, DeepseekV32IndexerBackend, KpoolTailBackend,
                                                   get_mamba_attn_backend(MambaAttentionBackendEnum.GDN_ATTN)])
        layout = resolve_kv_cache_layout(cfg, [[value.name for value in supported]], specs.values())
        require(layout.name == "LBHNC", "unexpected physical cache layout")
        record({"event": "normalized", "attention_block_tokens": cfg.cache_config.block_size,
                "mamba_block_tokens": cfg.cache_config.mamba_block_size, "mamba_shapes": shapes,
                "mamba_dtypes": [str(v) for v in dtypes], "layout": layout.name,
                "groups": [{"layers": g.layer_names, "spec": repr(g.kv_cache_spec)} for g in groups],
                "shared_pool_blocks": worker.num_blocks})
        torch.cuda.reset_peak_memory_stats()
        views = allocate_kv_cache(worker, torch.device("cuda"), layout, kernel_block_sizes=[64, 4, 262144, 262144, 262144, 262144])
        torch.cuda.synchronize()
        require(set(views) == set(specs), "missing allocated layer views")
        storage = {v.untyped_storage().data_ptr(): v.untyped_storage().nbytes() for v in views.values()}
        require(len(storage) == 1 and sum(storage.values()) == expected["unique_backing_bytes"], "unique backing allocation mismatch")
        record({"event": "allocated_and_zeroed", "unique_backing_bytes": sum(storage.values()), "layer_views": len(views),
                "cuda_memory_allocated": torch.cuda.memory_allocated(), "cuda_memory_reserved": torch.cuda.memory_reserved(),
                "cuda_peak_allocated": torch.cuda.max_memory_allocated()})
        scheduler = generate_scheduler_kv_cache_config([worker])
        scheduler_bs, hash_bs = resolve_kv_cache_block_sizes(scheduler, cfg)
        require(scheduler_bs == hash_bs == 4456448, "scheduler/hash normalization mismatch")
        require(cfg.scheduler_config.watermark == 0, "nonzero reservation watermark")
        record({"event": "scheduler_normalized", "scheduler_block_tokens": scheduler_bs, "hash_block_tokens": hash_bs})
        manager = KVCacheManager(scheduler, max_model_len=262144, scheduler_block_size=scheduler_bs,
                                 hash_block_size=hash_bs, max_in_flight_tokens=cfg.max_in_flight_tokens,
                                 enable_caching=False, use_eagle=False, watermark=cfg.scheduler_config.watermark)
        requests = [Request(request_id=f"cache-preflight-{i}", prompt_token_ids=[1] * 262144,
                            sampling_params=SamplingParams(max_tokens=1), pooling_params=None) for i in range(5)]
        reserve_four(manager, requests, seed, record)
        torch.cuda.synchronize()


def main():
    if sys.flags.optimize:
        raise SystemExit("optimized Python is forbidden")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    require(0 <= args.seed < 2**64, "unsigned 64-bit public seed required")
    decision_path = ROOT / "configs/decision-specs/glm53-cache-preflight.json"
    decision = strict_json(decision_path)
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {"qualification": decision["qualification"], "seed": args.seed, "scorer_sha256": sha256_file(Path(__file__)),
                "decision": {"sha256": sha256_file(decision_path)}, "binary_sha256": sha256_file(Path(sys.executable).resolve()),
                "metadata": {p.name: {"sha256": sha256_file(p)} for p in args.metadata.iterdir() if p.is_file()}, "start_unix": time.time()}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with (args.output / "raw.jsonl").open("w") as raw, (args.output / "traceback.log").open("w") as error_log:
        def record(row):
            raw.write(json.dumps({"time_unix": time.time(), **row}, allow_nan=False) + "\n")
            raw.flush()
        try:
            run_native(args.metadata, decision["expected"], args.seed, record)
            summary = {"verdict": "PASS"}
        except Exception as error:
            traceback.print_exc(file=error_log)
            record({"event": "failure", "failure": repr(error)})
            summary = {"verdict": "FAIL", "failure": repr(error)}
    summary.update(qualification=decision["qualification"], model_loaded=False, actual_input_tokens_processed=0,
                   context_capability="not measured", performance="not measured", raw_sha256=sha256_file(args.output / "raw.jsonl"))
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))
    raise SystemExit(0 if summary["verdict"] == "PASS" else 1)


if __name__ == "__main__":
    main()
