# Local model upgrade QA record

Date: 2026-07-17

## Scope and decision

The VitalsSight language layer now defaults to `qwen3.6:35b` through Ollama. This is the quality-first local profile validated on the current workstation, not a claim that one open-weight model is universally best. The deterministic measurement, candidate selector, release/review/retake gate, report data and audit log remain authoritative. The language model only selects read-only tools, explains recorded evidence and prepares explicitly confirmable workflow actions.

The selected Ollama model reports 36.0B total parameters, Q4_K_M quantization, a 262,144-token architectural context limit, and completion, vision, tool and thinking capabilities. The deployed application intentionally uses a smaller 8,192-token context and unloads the model after each answer to preserve memory for the UI, API and multimodal sidecars.

Primary references:

- [Qwen3.6-35B-A3B official model card](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
- [Qwen3.6 official Ollama tags](https://ollama.com/library/qwen3.6/tags)
- [Ollama thinking capability](https://docs.ollama.com/capabilities/thinking)
- [Ollama structured outputs](https://docs.ollama.com/capabilities/structured-outputs)
- [Ollama tool calling](https://docs.ollama.com/capabilities/tool-calling)

## Workstation envelope

- CPU: AMD Ryzen 7 5800HS, 8 cores / 16 threads
- System memory: approximately 39.4 GB
- Inference processor reported by Ollama: 100% CPU
- Installed model size: approximately 23 GB
- Memory after loading the model with an 8K context: approximately 1.2 GB free during the four-case stress run
- Memory after `keep_alive=0` unloaded the model: approximately 25.5 GB free with the UI and API still running

Keeping the model resident was therefore rejected for this workstation. The default `VITALSSIGHT_ASSISTANT_KEEP_ALIVE=0` is a stability requirement, not a model-quality setting.

## Alternatives reviewed

| Candidate | Local package | Strength relevant to VitalsSight | Reason not selected as the default |
|---|---:|---|---|
| Qwen3.6-35B-A3B | 23 GB Q4 | Current multilingual, multimodal MoE; tools, thinking and structured output in one model | Selected, with per-request unloading because the resident memory margin is too small |
| Qwen3.5-27B | 17 GB Q4 | Current multilingual multimodal dense model | All 27B parameters are active per token, making CPU interaction less practical than the selected MoE profile |
| GLM-4.7-Flash | 19 GB Q4 | Chinese/English 30B-A3B reasoning and agentic model | Text-only in the local Ollama package and would still require a separate vision sidecar; retained as a future server/GPU candidate |
| GPT-OSS-20B | 14 GB MXFP4 | Strong reasoning, tools and structured output with 3.6B active parameters | Official model card describes mostly English, text-only training; weaker fit for this bilingual multimodal workflow |
| DeepSeek-R1 32B | 20 GB Q4 | Strong text reasoning | Dense, text-only profile with a less favorable local CPU and multimodal fit |

Official comparison sources: [Qwen3.5 Ollama tags](https://ollama.com/library/qwen3.5/tags), [GLM-4.7-Flash model card](https://huggingface.co/zai-org/GLM-4.7-Flash), [GPT-OSS launch and architecture](https://openai.com/index/introducing-gpt-oss/), [GPT-OSS model card](https://openai.com/index/gpt-oss-model-card/), and [DeepSeek-R1 Ollama registry](https://registry.ollama.com/library/deepseek-r1).

## Measured behavior

These are finite workstation checks, not standardized model benchmarks. Prompt contract revisions mean the rows are not a leaderboard and should not be compared as if every token budget were identical.

| Check | Result |
|---|---|
| Legacy `qwen3:4b`, eight sampled cases | 8/8 accepted; mean 63.20 s; maximum 105.28 s |
| `qwen3:8b` explicit-thinking smoke | 1/1 accepted; 270.16 s |
| Qwen3.6 four-case bilingual state sample, direct mode | 4/4 safe outcomes; mean 90.83 s; maximum 104.63 s; one pre-fix JSON truncation used deterministic fallback |
| Qwen3.6 Chinese release case after output-budget fix | 1/1 live-model response; 140.79 s; no fallback |
| Qwen3.6 review case through the running FastAPI product | Live model; 121.47 s; `review`; HR withheld; validation passed |
| Qwen3.6 explicit-thinking review case | 1/1 accepted; 406.62 s; no material acceptance improvement over direct mode |
| Qwen3.6 image analysis experiment | 135.07 s and boundary-compliant, but too slow to replace the validated 4B image sidecar |

Explicit thinking is therefore opt-in rather than the local default. The stronger model still performs direct schema-constrained composition; disabling the separate reasoning trace does not bypass evidence retrieval, structured output, post-validation or the deterministic fallback.

## Output-contract hardening

The model must now return four schema fields:

1. `direct_answer`
2. `evidence_explanation`
3. `next_step`
4. `used_evidence_ids`

Selected-case answers are rejected if either the evidence explanation or verification step is empty. The public answer is assembled into three visible paragraphs, and the evaluation harness now fails a live selected-case answer that omits one of these sections. Tool routing remains short and non-thinking. A malformed, truncated, contradictory, clinically overreaching or numerically unsupported answer falls back to deterministic evidence guidance.

## Validated default configuration

| Setting | Value |
|---|---|
| Text model | `qwen3.6:35b` |
| Vision sidecar | `qwen3-vl:4b-instruct` |
| Speech sidecar | `faster-whisper small`, CPU int8 |
| Context | 8,192 tokens |
| Structured answer budget | 768 tokens |
| Explicit thinking | disabled by default; opt-in |
| Text keep-alive | `0` (unload after response) |
| Vision keep-alive | `0` (unload after analysis) |
| Provider timeout | 300 seconds |

## Verification boundary

The code change passed 91 automated Python tests, JavaScript syntax checks and `git diff --check` before final browser conformance. The model checks establish finite technical behavior on curated research-workflow cases. They do not establish clinical validity, medical-device performance, production security, end-to-end real-time operation or usability in a clinical environment.
