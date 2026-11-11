# Local ChatCV quality baseline

The reference set contains 20 questions about `CV_Demo.pdf`. It has 12 fully
answerable, four partially answerable and four unanswerable questions. Each
expected passage records its PDF page.

Run these commands from the repository root with the locked Pixi environment.
Use a running local Ollama server with `gpt-oss:20b` and `all-minilm:22m`
installed. Start MLflow in a separate terminal if it is not already running.

```sh
MLFLOW_DISABLE_TELEMETRY=true pixi run --as-is mlflow server --host 127.0.0.1 --port 5000
```

Generate the answers and populate the review queue.

```sh
pixi run --as-is python -m experiments.run_reference_case --prepare-review \
  --chat-model gpt-oss:20b --embedding-model all-minilm:22m
```

The runner reuses successful traces when the reference case and recorded
configuration match. Failed traces remain visible and a repeat run can retry
their cases. Use `--experiment` with a new name to capture a separate run.

Open the printed experiment URL and select **Review → ChatCV baseline review**.
Compare the answer and retrieved passages with the numbered expected facts. The
queue uses the same number of fact slots for every case. Leave slots without a
numbered expected fact blank. The export records those slots as `n/a`.

| Field | Values |
| --- | --- |
| Retrieval by fact | Select `found` or `missing` for each expected fact. |
| Answer by fact | Select `correct`, `missing` or `incorrect` for each expected fact. |
| Unsupported claims | Select `none` or `present`. Use the optional comment to describe any claims. |
| Information gaps | Leave blank or select `n/a` for fully answerable cases. Select `appropriate` or `inappropriate` for the other cases. |

Save the human review before viewing judge results. Export the cases and validate the labels with the
following command.

```sh
pixi run --as-is python -m experiments.run_reference_case \
  --export-review .codex-local/exports/chatcv-baseline.json
```

The export lists missing cases and invalid or missing labels alongside the
original judgments. MLflow's database, artifacts and private review exports
are Git-ignored.

## Compare local judges

Run the comparison after the human export passes validation. Use two models
already installed in the local Ollama server.

```sh
pixi run --as-is python -m experiments.compare_reference_judges \
  .codex-local/exports/chatcv-baseline.json \
  .codex-local/exports/chatcv-judge-comparison.json \
  --model gpt-oss:20b --model gemma4:e4b
```

Open **ChatCV - Local Judge Comparison** in MLflow for the model runs and
their evaluation traces. The JSON report links each result to the saved
ChatCV trace and records the input export hash, judge model digests, prompts,
settings and agreement counts. Retrieval sees the passages returned by
ChatCV. Answer judgments use the saved answer and the reference resume.
The comparison applies the human labels after the model has responded.
