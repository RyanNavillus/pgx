# AlphaZero example

A simple (Gumbel) AlphaZero [[Silver+18](https://www.science.org/doi/10.1126/science.aar6404), [Danihelka+22](https://openreview.net/forum?id=bERaNdoegnO)] example using [Mctx](https://github.com/deepmind/mctx) library. See [Pgx paper](https://openreview.net/forum?id=UvX8QfhfUx) for more details.

![](assets/pgx-az-training.png)

> [!NOTE]
> This implementation of AlphaZero demonstrates sufficient learning performance in environments including 9x9 Go, but it has some slight differences in learning details compared to the original AlphaZero and Gumbel AlphaZero. An implementation that addresses these differences and focuses on enhanced efficiency is currently under development and is expected to be released shortly.

## Usage

Note that you need to install `jax` and `jaxlib` in addition to the packages written in `requirements.txt` according to your execution environment.

```sh
$ pip install -U pip && pip install -r requirements.txt
$ python3 train.py env_id=go_9x9 seed=0
```

`fix_autoreset_state` defaults to `true` and clears terminal metadata from the
reset state carried into the next self-play step. Set
`fix_autoreset_state=false` to disable this fix.

## Rendering checkpoints

Render every checkpoint in a training run, with one self-play episode per
checkpoint, using:

```sh
$ python3 render.py checkpoints/go_9x9_20260714104146
```

The default output directory is
`renders/go_9x9_20260714104146`. PNG frames are named like
`000000_0_0.png` and GIFs like `000000_0.gif`; episode and step numbers are
zero-based. Use `--episodes 3` for three episodes per checkpoint, `--gif`
or `--no-gif` to control GIF creation, `--output-dir PATH` to choose a
different output location, `--max-steps N` to override the checkpoint
rollout horizon, and `--debug` to print JAX step progress.

## Reference

- [[Silver+18](https://www.science.org/doi/10.1126/science.aar6404)] "A general reinforcement learning algorithm that masters
chess, shogi, and go through self-play"
- [[Danihelka+22](https://openreview.net/forum?id=bERaNdoegnO)] "Policy improvement by planning with Gumbel"


## Change history

- **[#1107](https://github.com/sotetsuk/pgx/pull/1107)** Extract `compute_loss_input` ([wandb report](https://api.wandb.ai/links/sotetsuk/979hmps8)).
- **[#1106](https://github.com/sotetsuk/pgx/pull/1106)** Use `optax.softmax_cross_entropy` ([wandb report](https://api.wandb.ai/links/sotetsuk/8w0or84k)).
- **[#1088](https://github.com/sotetsuk/pgx/pull/1088)** Adjust to API v2 ([wandb report](https://api.wandb.ai/links/sotetsuk/0g44pjsg)).
- **[#1055](https://github.com/sotetsuk/pgx/pull/1055)** Use default Gumbel AlphaZero hyperparameters ([wandb report](https://api.wandb.ai/links/sotetsuk/o8752t54)).
- **[#1026](https://github.com/sotetsuk/pgx/pull/1026)** Initial version. Supposed to reproduce the [Pgx paper](https://openreview.net/forum?id=UvX8QfhfUx) results ([wandb report](https://api.wandb.ai/links/sotetsuk/5q30e5n9)).
