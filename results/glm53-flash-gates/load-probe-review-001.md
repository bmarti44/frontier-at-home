# Component-load review candidate1 / campaign40

Candidate9f37a677, audit eb214abc. Adversarial reviewer: two HIGH findings;
no separate loader/API high issue verified. Gap reviewer independently reproduced
H1; final attestation pending. No GPU attempt has used this candidate.

H1:42_probe_glm53_load.py validate_memory accepts all CUDA counters zero despite
1,268,776,960 bytes of declared ordinary storage. Retained GTensorCache geometry
is unchecked; MoE zero concurrency and empty scratch pass. Require phase-specific
current/peak lower bounds from live storage and required overlap, exact retained
cache/scratch coverage and valid storage relationships.

H2:score_capture accepts a20-byte invalid safetensors file while reporting1.27GB
checked when its child-authored inventory matches that file. The reviewer used
real independently computed seed7 ordinary and excluded tensor digests totaling
1,335,894,016 bytes. Require independently reconstructed canonical complete-file
size/hash, including header and excluded payload, against actual retained bytes.

Candidate2 is limited to closing H1/H2. Previously frozen helpers stay unchanged.
