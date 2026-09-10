# Direct context replay: output limit failure

All four streams verified 250,128 input tokens each (1,000,512 aggregate) and
generated 256 timestamped output tokens each, with overlapping generation and
zero preemptions. The fixed scorer returned FAIL: every output was reasoning,
finished at the length limit, and contained no final answer. Reasoning is not
substituted for final retrieval or negative-control acceptance.

This diagnostic used CUDA_LAUNCH_BLOCKING=1 and exact old inputs. The original
failed run also had a prior short request which this replay did not have; success
at processing input cannot be attributed to synchronization alone. Eight tail-seed
kernel files were generated after freeze at decode, and usage_stats changed.
The generated files are preserved separately in generated-cache.tar.gz. A new
prepared-cache freeze is required before confirmation. Original
freeze coverage omissions are retained and do not support full qualification.

The model was stopped through its verified systemd invocation after scoring.
No model process survived; available memory recovered above 110 GiB. Guard
BrokenPipe and semaphore warnings during intentional shutdown remain preserved.
The raw kernel journal for the attempt contains no entries.

The split archive preserves all client rows and unfiltered metrics, server and
watchdog logs, process identity, freeze inventories, scorer sources and stop
observations. Join parts in order before extraction:

```bash
cat attempt.tar.gz.part-* | tar -xz
```

Next bounded change: preregister max_tokens=2048 in the request configuration,
keep prompt construction and existing retrieval scorer unchanged, and prepare
fresh fixtures after a new freeze/public seed. The named profile also removes
CUDA_LAUNCH_BLOCKING=1; this is a new native-profile confirmation, not an
output-budget-only replay. Execute through the actual named
full-context profile and preserve its readiness request history. This remains
a failed capability attempt; no qualified production speed is published.
