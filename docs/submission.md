# Submission — tokensand.com/vastnyc/submit

## Team

- Coleman Isner — colemanisner@gmail.com
- <<name/email>>

## Description (~150 words)

Other tools find what your video archive is missing. Spore grows the data to fill it, and shows where your AI goes blind. Spore is a video agent that audits your archive: it searches VSS for what your cameras have never seen, generates that missing footage, and measures what breaks in your own pipeline.

Across 612 real clips searched with VSS, highway cameras have zero rain, fog, snow, glare, or night-rain clips, and no camera has snow. A W&B Inference agent writes the gap report, traced in Weave. Spore fills the gap with a geometry-preserving physics weather layer (99 clips) and NVIDIA Cosmos Transfer 2.5 relighting, measured against a clear-day control. It then runs the VSS Detector (YOLO11s) and Reasoner (Cosmos3) on the results. Detector recall falls below 50% at snow 0.34, fog 0.35, and rain 0.60, while Cosmos Reason keeps counting vehicles. Real-archive confirmation is weak: glare is the strongest case, fog is unconfirmed, and no real snow exists. A synthetic fine-tune lost to a free contrast trick. At this scale, synthetic footage works better as a stress test than as training data.

## Tools used

- **VAST**: VSS `/api/v1/search`, explore, and captions, used to find the coverage gap and to find real-condition clips for confirmation
- **NVIDIA**: Cosmos Transfer 2.5 (Modal H100), hosted Cosmos3 Nano Reasoner, and hosted YOLO11s
- **CoreWeave / Weights & Biases**: serverless W&B Inference for the gap-report agent, plus Weave traces; the hosted models run on CoreWeave GPUs
- **SpaceXAI / Cursor**: built with Cursor agents, including Grok 4.7 running parallel tasks
- Also used: Modal (H100 for Cosmos Transfer, L40S for the fine-tune experiment), Ultralytics, OpenCV, React + Vite UI

## Links

- Repo: https://github.com/ColemanIsner/team16
- Demo video: <<add link>>
- Weave: https://wandb.ai/colemanisner-sporelabs/sporelabs-hackathon/weave
