# Submission — tokensand.com/vastnyc/submit

## Team

- Coleman Isner — colemanisner@gmail.com
- <<name/email>>

## One-sentence description

Spore helps video model operators find gaps in their evaluation data, grow synthetic data to cover them, and build new evals and benchmarks for training better models.

## Description (~150 words)

Video AI fails quietly on conditions your cameras don't see often. Spore finds those gaps, grows synthetic data to cover them, and turns that data into evals, in a loop. It searches VAST's video index for missing conditions: the highway cameras have 30 clips, mostly clear daylight, and none at night, in rain, in heavy fog or in snow. An LLM on W&B Inference, traced in Weave, ranks which gaps matter. Spore grows the missing weather onto real clips with a physics weather layer and NVIDIA Cosmos Transfer. The cars stay in place, so the labels carry over and testing runs automatically. On CoreWeave GPUs, it tests the models in the VSS pipeline: in light fog, YOLO finds only four in ten visible cars, while Cosmos Reason keeps counting them. The result is a new benchmark and training set for better models.

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
