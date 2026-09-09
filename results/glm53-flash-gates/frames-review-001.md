# Frame gate review

Candidate 1: `cccbf01c`. Campaign review round 4. Both persistent reviewers
found high F1; the processor-method cap misses normal serving ingestion.

Pinned `Glm5NextVideoBackend.compute_frames_index_to_sample` calls shared
`glm_sample_frame_indices` directly, ignoring `target.num_frames=16`.
Reviewers executed this method and obtained 120 frames from a 60-second clip
and 1,200 from a 600-second clip. `create_hf_metadata` sets `do_sample_frames`
false, and the processor then skips the patched method. Existing fallback
sampler tests passed but did not cover this route.

Focused acceptance: move the cap into the shared helper, test the actual
loader method and metadata path as well as processor fallback, preserve
within-cap sample indices, and declare `max_frames=16` in video I/O settings.
The JPEG-frame URI route already slices to configured `num_frames`; the
frozen admission middleware blocks request overrides. No second high finding.
