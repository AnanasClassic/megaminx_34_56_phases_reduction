# Pair56 Q-MLP training

This is the first model stage after the exhaustive M1--M4 gate. It trains one
Q-model directly on the combined quotient `G7 -> G9`, not two sequential phase
models and not a boundary-repair model.

## Fixed problem

- source faces: `U R F L BR`;
- target subgroup faces: `U R F`;
- metric: FTM, four non-identity powers per face;
- actions: 20;
- state: p900 S120 with all 54 stickers movable by `G9` collapsed to one class;
- fixed sticker classes: 66 plus one black class, hence 67 classes;
- exact coordinate-space size: 1,664,617,163,489,280;
- symmetry augmentation: identity only;
- random-walk `K_max`: **hard-coded to 26** and not exposed as a CLI option.

The tracked `training/pair56/problem.json` is the repo-local proof input.
Without the historical p900 file, `scripts/audit_phase_groups --pair pair56`
recomputes the exact orders of `G7`, `G8`, and `G9`, both individual quotient
factors, and the pointwise target stabilizer from the committed actions.
Loading the manifest also checks every action against `FullStateV1` through the
stored conjugacy.

The manifest was originally derived from the pinned p900 generator with SHA-256
`3e38e75ee4f3387c33917393068b2fadf7959b3490f86d6e924f266960f45dbd`.
That file is historical construction provenance, not a proof-time dependency.
If it is separately available, regenerate the manifest with:

```bash
scripts/build_pair56_problem
```

Generation uses Schreier--Sims to verify both the subgroup index and the exact
target stabilizer. The pointwise stabilizer of the 66 retained sticker
positions has the same order as `<U,R,F>`, while
`|<U,R,F,L,BR>| / |<U,R,F>|` equals the declared coordinate-space size. Thus
the collapse represents exactly `G7/G9`, not a coarser coloring quotient.

The p900 base-turn direction is inverse to the `FullStateV1` convention. The
problem builder proves a simultaneous 120-sticker conjugacy for all twelve
faces and stores it in the manifest. Its four p900 powers are therefore emitted
in order `4,3,2,1`, so action names such as `U1` and `BR4` can be replayed
directly by the independent verifier without another power conversion.

The sampler rejects moves that are self-loops in the quotient. This matters at
the target, where all twelve `U/R/F` turns are legal FTM moves but leave the
collapsed target unchanged. Consecutive moves of the same face are also
excluded because they reduce to one FTM move or identity.

## Model

The default model is a 802,260-parameter Q-MLP:

```text
position/class embedding-bag: 120 * 67 -> 64
hidden:                         64 -> 256
residual blocks:                2 * (256 -> 256 -> 256)
Q head:                         256 -> 20
```

It uses the identity-only form of the random-walk sparse-Q pipeline used for
model `1783930119`: at a random trajectory pivot, the inverse predecessor move
is labelled `depth-1` and the sampled successor is labelled `depth+1`.

## Smoke test

This runs two tiny CPU epochs, writes only to `/tmp`, reloads the checkpoint and
checks validation inference:

```bash
scripts/train_pair56 \
  --smoke --device cpu --amp fp32 \
  --output-dir /tmp/mdr-pair56-smoke --run-id smoke

scripts/test_pair56 \
  --checkpoint /tmp/mdr-pair56-smoke/pair56-qmlp_smoke_best.pt \
  --device cpu --amp fp32 --tests 0 \
  --val-size 128 --val-batch-size 64
```

An untrained smoke checkpoint can also exercise complete shallow beam replay;
the large beam is intentional because the model has seen only eight batches:

```bash
scripts/test_pair56 \
  --checkpoint /tmp/mdr-pair56-smoke/pair56-qmlp_smoke_best.pt \
  --device cpu --amp fp32 \
  --tests 5 --scramble-depth 2 --beam-width 1024 --max-steps 2 \
  --val-size 64 --val-batch-size 64 --require-all
```

## Full training

The default batch is 1,024. This is close to the old transformer's effective
batch after its ten symmetry rows per 64 base states, while using independent
quotient walks here. `K_max` is omitted because it cannot be changed from 26.

```bash
scripts/train_pair56 \
  --device cuda:0 --amp bf16 \
  --epochs 8192 --steps-per-epoch 256 --batch-size 1024 \
  --K-min 2 \
  --hd1 64 --hd2 256 --residual-blocks 2 \
  --lr 1e-4 --weight-decay 0.003 --grad-clip 1 \
  --val-size 16384 --val-batch-size 2048 \
  --val-every 32 --log-every 1 --save-every 100 \
  --run-id baseline
```

On the local A100 benchmark, the optimized sampler plus model processes about
90,000 states/s at batch 1,024. An epoch contains 262,144 independently sampled
states and takes roughly three seconds when the GPU is otherwise available.
The former batch-64 rejection implementation took about 8.3 seconds for only
16,384 states. Epoch lines are printed every epoch; validation remains every
32 epochs.

Checkpoints, CSV logs and model metadata are written under `models/pair56/`.
Training data is generated on the fly, so the run does not create a large
dataset.

## Testing a trained model

First measure held-out sparse-Q metrics and beam-search success on 1,000 fresh
depth-26 walks:

```bash
scripts/test_pair56 \
  --checkpoint models/pair56/pair56-qmlp_baseline_best.pt \
  --device cuda:0 --amp bf16 \
  --tests 1000 --scramble-depth 26 \
  --beam-width 256 --search-batch-size 32 --max-steps 26 \
  --val-size 16384 --val-batch-size 2048
```

Search batches independent roots into one GPU model call per depth. On the
local A100, 100 depth-100 walks at beam 256 completed at 9.7 states/s versus
0.44 states/s for the former sequential implementation. Deduplication remains
exact per root, and every returned word is replayed from its initial state.
Candidate transfer is adaptive: the evaluator first materializes the best
`2 * beam_width` rows and falls back to the rest of the identically ordered
beam only for roots that still need entries after exact deduplication. On a
fixed beam-128 hard-state benchmark this improved throughput from 16.35 to
53.16 states/s without changing the 83/100 solved set.

The first 100 unresolved physical pair56 representatives can be tested at the
actual proposed bound with:

```bash
scripts/test_pair56 \
  --checkpoint models/pair56/pair56-qmlp_baseline-fast_best.pt \
  --device cuda:0 --amp bf16 \
  --hard-states 100 --hard-offset 0 \
  --beam-width 2048 --search-batch-size 32 --max-steps 25 \
  --val-size 16384 --val-batch-size 2048
```

Hard-state IDs come from `reductions/pair56/remaining_ids.bin`; physical states
come from the composition artifact. Returned words are checked both in the
model quotient and independently by applying verifier moves to `FullStateV1`
and testing membership in `G9`. The summary reports every unsolved hard ID.

## Resumable certificate pass

Freeze a checkpoint, then scan the 399,167 unresolved representatives in
bounded-memory chunks. The SQLite database records the exact solution word,
length, state checksum, checkpoint hash and epoch, and the beam width at which
the solution was found. Existing certified IDs are skipped on reruns.

```bash
cp models/pair56/pair56-qmlp_baseline-fast_best.pt \
   models/pair56/pair56-qmlp_beam64-snapshot.pt

total=399167
chunk=10000
database=certificates/pair56/beam-cascade.sqlite3

for ((offset=0; offset<total; offset+=chunk)); do
  count=$((total - offset))
  ((count > chunk)) && count=$chunk

  scripts/test_pair56 \
    --checkpoint models/pair56/pair56-qmlp_beam64-snapshot.pt \
    --device cuda:0 --amp bf16 \
    --hard-states "$count" --hard-offset "$offset" \
    --beam-width 64 --search-batch-size 32 --max-steps 25 \
    --val-size 128 --val-batch-size 128 \
    --solutions-db "$database" \
  | jq -r '[.hard_states_tested, .solved, .solved_new, .certificate_store.certificates] | @tsv'
done | awk '{
  tested += $1
  covered += $2
  printf "tested=%d covered=%d rate=%.6f new=%d db_total=%d\n",
         tested, covered, covered/tested, $3, $4
  fflush()
}'
```

Independently replay every stored certificate without loading the model:

```bash
scripts/verify_pair56_solutions \
  --database certificates/pair56/beam-cascade.sqlite3
```

The database is partial until its verified count reaches 399,167; the 8,461
states already closed by exact local rewrites remain in the separate M4
reduction certificate.

Add `--require-all` only when the selected budget is expected to close the full
test set. Every returned word is replayed against the exact collapsed target.
Random-walk evaluation remains diagnostic. Only hard-state solutions persisted
in the certificate database contribute to exhaustive pair56 coverage.
