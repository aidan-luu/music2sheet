# Training

Compute target per model head:

- **Hardware**: 1 x NVIDIA A100 80GB (H100 acceptable, faster). Single-node only;
  no multi-node loops are planned for this milestone.
- **Wall time**: ~1 week per head (melody PR-4/5, chord PR-6/7, key PR-8,
  voicing PR-9). Scale-up runs may extend to 2 weeks.
- **Storage**: ~500 GB scratch for dataset caches + checkpoints; final
  artifacts published to HuggingFace Hub via `ml/models/registry.yaml`.

## Kill criterion (PR-5)

If melody F1 on the SheetSage held-out test set falls below **0.80** at the
end of PR-5 (melody scale-up), Agent C must set state to `failed` with
`reason: "kill-criterion"`. The orchestrator pauses the team and consults
the user before PR-6 (chord head) begins. See `.orchestrator/dependency_graph.yaml`
hard rule `kill-criterion-gate`.

## Run layout

Each training job writes to `ml/training/runs/<run-id>/`:

- `config.yaml`      - hyperparameters + dataset manifest
- `eval.json`        - final metrics (consumed by Agent D's leaderboard)
- `checkpoints/*.pt` - intermediate + final weights

## Links

- Project plan: `../../skills.md` (Agent C section)
- Dependency DAG: `../../.orchestrator/dependency_graph.yaml`
- Eval harness owner: Agent D, under `../../evals/`
