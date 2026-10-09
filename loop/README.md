# Spore loop — W&B Inference + Weave

LLM calls for the failure-seeking loop go through [W&B Inference](https://docs.wandb.ai/inference) (OpenAI-compatible, hosted on CoreWeave). Each call can be traced with [Weave](https://docs.wandb.ai/weave). If `WANDB_API_KEY` is unset, tracing is a no-op and `propose_next_condition` returns a deterministic rationale so the loop still runs.

## Setup

```bash
uv venv loop/.venv
uv pip install -r loop/requirements.txt --python loop/.venv/bin/python
```

Put secrets in the repo-root `.env` (loaded automatically) or export them in the shell. Import from the repo root:

```python
import loop.tracing as tracing
import loop.llm as llm

tracing.init()  # also called when loop.llm is imported
print(llm.propose_next_condition({"arms": [], "budget_used": 0}, conditions))
```

## Environment

| Variable | Required | Purpose |
|---|---|---|
| `WANDB_API_KEY` | for live calls | W&B API key. Unset → Weave no-op, `chat` returns `""`, `propose_next_condition` uses the deterministic fallback. |
| `WANDB_ENTITY` | no | W&B team name. With `WANDB_PROJECT`, sent as the `OpenAI-Project` header `entity/project` ([docs](https://docs.wandb.ai/inference)). |
| `WANDB_PROJECT` | no | Weave project and the project half of the inference header. Default `sporelabs-hackathon`. |
| `WANDB_INFERENCE_PROJECT` | no | Overrides the inference project header outright (`team/project`). |
| `SPORE_LLM_MODEL` | no | Model id override. Default `deepseek-ai/DeepSeek-V4-Pro-0813`. |

`tracing.init(project="sporelabs-hackathon")` calls `weave.init` only when `WANDB_API_KEY` is set. `tracing.traced` is `weave.op` after a successful init and an identity decorator otherwise. Init never raises when W&B is not configured.

## API

- `llm.chat(messages, model=None, json_mode=False) -> str` — chat completion. `json_mode=True` sets `response_format={"type": "json_object"}`.
- `llm.propose_next_condition(failure_map: dict, conditions: list) -> dict` — returns `{"condition": ..., "rationale": ...}` for the demo. Thompson sampling owns the numeric arm; this text is for display. Without a key the rationale is deterministic: highest observed `failure_rate`, otherwise the first condition that has no arm yet.

## Default model and candidates

Default is **`deepseek-ai/DeepSeek-V4-Pro-0813`** (DeepSeek V4-Pro, 49B active / 1.6T total), the strongest generally available reasoning model on W&B Inference as of the [available models](https://docs.wandb.ai/inference/models) list.

Other strong open candidates on that list (pass as `model=` or `SPORE_LLM_MODEL`):

| Model id | Why you might switch |
|---|---|
| `deepseek-ai/DeepSeek-V4-Pro-0813` | Default. Best reasoning / agentic quality. |
| `deepseek-ai/DeepSeek-V4.1-Flash` | Faster multimodal MoE, still strong at coding and agents. |
| `deepseek-ai/DeepSeek-V3.1` | Hybrid thinking / non-thinking DeepSeek. |
| `zai-org/GLM-5.2` | 40B/744B MoE, long context. |
| `moonshotai/Kimi-K2.6` | 32B/1T multimodal MoE. |
| `openai/gpt-oss-120b` | High-reasoning MoE, lower active-parameter cost. |
| `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B` | Long-running agent workloads. |
| `meta-llama/Llama-3.3-70B-Instruct` | Dense 70B instruction follower; simpler outputs. |
| `Qwen/Qwen3.8-27B` | Smaller dense model when latency matters more than depth. |
