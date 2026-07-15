# Pair34 Q-MLP training

This model is trained directly on the combined `G5/G7` quotient for the
phase-3 plus phase-4 target. It is separate from pair56.

## Certified problem

- source faces: `U R F L BR BL FR`;
- target subgroup faces: `U R F L BR`;
- metric: FTM, four powers per face;
- actions: 28;
- exact quotient size: 14,234,011,545,600;
- state: 42 fixed p900-S120 sticker classes plus one black class;
- random-walk `K_max`: fixed to 22;
- target proof length for maximal pair compositions: 21.

Regenerate `training/pair34/problem.json` with:

```bash
scripts/build_pair34_problem
```

The builder checks the exact subgroup index, equality of the pointwise coloring
stabilizer with `G7`, and simultaneous move conjugacy with `FullStateV1`.

## Smoke test

```bash
scripts/train_pair34 \
  --smoke --device cpu --amp fp32 \
  --output-dir /tmp/mdr-pair34-smoke --run-id smoke
```

## Production training

```bash
scripts/train_pair34 \
  --device cuda:0 --amp bf16 \
  --epochs 8192 --steps-per-epoch 256 --batch-size 1024 \
  --K-min 2 \
  --hd1 64 --hd2 256 --residual-blocks 2 \
  --lr 1e-4 --weight-decay 0.003 --grad-clip 1 \
  --val-size 16384 --val-batch-size 2048 \
  --val-every 32 --log-every 1 --save-every 100 \
  --run-id baseline
```

The default model has 619,996 parameters. Training data is generated on the
GPU and does not create a dataset on disk. A measured A100 epoch contains
262,144 states and takes about 3.3 seconds when the device is otherwise idle.
Checkpoints and logs are written to `models/pair34/`.

## Random-walk evaluation

Measure beam-32 success on 100 fresh depth-100 walks:

```bash
scripts/test_pair34 \
  --checkpoint models/pair34/pair34-qmlp_baseline_best.pt \
  --device cuda:0 --amp bf16 \
  --tests 100 --scramble-depth 100 \
  --beam-width 32 --search-batch-size 100 --max-steps 22 \
  --val-size 1024 --val-batch-size 1024
```

Every returned word is replayed in the certified `G5/G7` coordinate. This is a
diagnostic random-walk test, not yet the exhaustive 536,369 hard-state pass.

Test the first 100 unresolved maximal pair34 compositions at the proposed
length bound:

```bash
scripts/test_pair34 \
  --checkpoint models/pair34/pair34-qmlp_baseline_best.pt \
  --device cuda:0 --amp bf16 \
  --hard-states 100 --hard-offset 0 \
  --beam-width 32 --search-batch-size 100 --max-steps 21 \
  --val-size 1024 --val-batch-size 1024
```

Hard-state solutions are additionally replayed on the physical `FullStateV1`
representative and must reach `G7`.

## Resumable beam-32 certificate pass

Freeze the checkpoint and scan the 536,369 unresolved representatives in
bounded-memory chunks. Existing certificates are skipped on reruns.

```bash
cp models/pair34/pair34-qmlp_baseline_best.pt \
   models/pair34/pair34-qmlp_beam32-snapshot.pt

total=536369
chunk=10000
database=certificates/pair34/beam-cascade.sqlite3

for ((offset=0; offset<total; offset+=chunk)); do
  count=$((total - offset))
  ((count > chunk)) && count=$chunk

  scripts/test_pair34 \
    --checkpoint models/pair34/pair34-qmlp_beam32-snapshot.pt \
    --device cuda:0 --amp bf16 \
    --hard-states "$count" --hard-offset "$offset" \
    --beam-width 32 --search-batch-size 128 --max-steps 21 \
    --val-size 128 --val-batch-size 128 \
    --solutions-db "$database"
done
```

Independently replay every persisted solution without the model:

```bash
scripts/verify_pair34_solutions \
  --database certificates/pair34/beam-cascade.sqlite3
```

Run the complete `32..4096` cascade unattended with:

```bash
scripts/run_pair34_beam_cascade \
  models/pair34/pair34-qmlp_night-snapshot.pt \
  certificates/pair34/beam-cascade.sqlite3
```

The runner is resumable and automatically uses smaller search batches for
larger beams. It stops early at full coverage and runs the independent verifier
after the final stage. `BEAMS`, `CHUNK`, `DEVICE`, `AMP`, and (for smoke tests)
`TOTAL` can be overridden through environment variables.
