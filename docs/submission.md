# Submission — tokensand.com/vastnyc/submit

## Description

Spore by SporeLabs takes real clips from the VAST index and grows structure-preserving synthetic variants with NVIDIA Cosmos Transfer. Fog, night, rain, and snow change the weather and the light while the camera, lanes, and vehicles stay put, so the stack's answer on the clean seed is the expected answer on every variant. A fidelity gate measures whether the generator kept the scene. Only then do YOLO detections and Cosmos Reason answers count as failures. A bandit spends the GPU budget on the conditions that break the stack. For each blind spot, VSS pulls real archive clips in that condition, and we check that the same failure shows up there. A condition-aware prompt is the fix, re-measured on held-out real clips. Weights & Biases Weave traces the run. The primary footage is real I-24 highway video, so the confirmation step is on real clips. Transfer runs on Modal. The loop's language-model calls use W&B Inference on CoreWeave.

## Tools used

- VAST (VSS / VastDB) — seed retrieval, real-archive confirmation search, ingest of synthetic clips
- NVIDIA — Cosmos Transfer 2.5 (generate), Cosmos Reason (summaries / Q&A under test), YOLO (detection under test)
- Weights & Biases — Weave traces of every gen/eval step and the eval table
- CoreWeave — W&B Inference for the loop agent's LLM calls; the VSS stack runs on CoreWeave GPUs
- SpaceXAI (Cursor) — Cursor agents, including Grok, building the system in parallel
