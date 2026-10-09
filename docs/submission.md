# Submission — tokensand.com/vastnyc/submit

## Team

- Coleman Isner — colemanisner@gmail.com
- <<name/email>>

## Description (~150 words)

Video AI fails quietly on the conditions your cameras don't see often. Spore finds where your video AI is weak, grows the data to cover it, and tests your models on it, in a loop. It reads all 612 real clips indexed in VAST's video search and searches for what is missing. The highway cameras have 30 clips, all in clear daylight: zero at night, in rain, in fog or in snow. The detector on those cameras has never been tested in any of it. An LLM on W&B Inference, traced in Weave, ranks which gaps matter. Spore grows the missing weather onto real clips with a physics weather layer and NVIDIA Cosmos Transfer. The cars stay where they were, so we already know the right answer. Then it tests the models inside the VSS pipeline, on CoreWeave GPUs. In light fog, YOLO finds only four in ten cars a person can still see, while Cosmos Reason keeps counting them. Then Spore picks the next weak spot and goes again.

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
