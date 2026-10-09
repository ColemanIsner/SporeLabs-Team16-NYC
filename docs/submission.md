# Submission — tokensand.com/vastnyc/submit

## Team

- Coleman Isner — colemanisner@gmail.com
- <<name/email>>

## Description (~150 words)

Spore finds where your video AI is weak and grows the data to fix it. It is a video agent that runs a six-step loop on your archive. It looks at every clip indexed in VAST VSS, along with its Cosmos Reason caption, and searches for the conditions your cameras have never seen. Across 612 real clips, the highway cameras had zero rain, fog, snow or glare footage, and no camera had snow. An LLM on W&B Inference, traced in Weave, ranks which gaps matter most. Spore then grows the missing weather onto real clips with NVIDIA Cosmos Transfer 2.5 and a physics weather layer that keeps every vehicle in place, so the labels stay exact. It tests the same models VSS runs, hosted YOLO11s and Cosmos Reason on CoreWeave GPUs, and finds the breaking point: detection drops below 50% at snow severity 0.34, fog 0.35 and rain 0.60. Then it picks the next weak spot and goes again.

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
