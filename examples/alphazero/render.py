"""Render self-play episodes from AlphaZero PGX checkpoints.

Example:

    python3 render.py checkpoints/go_9x9_20260714104146

Frames are named ``<checkpoint>_<episode>_<step>.png`` and GIFs are named
``<checkpoint>_<episode>.gif``. Episode and step numbers start at zero.
"""

import argparse
import io
import pickle
import time
from pathlib import Path

import cairosvg
import haiku as hk
import jax
import jax.numpy as jnp
import mctx
import numpy as np
import pgx
from PIL import Image

from network import AZNet


class _CheckpointConfig:
    """Compatibility target for Config objects saved by train.py."""
    env_id: pgx.EnvId = "go_9x9"
    seed: int = 0
    max_num_iters: int = 400
    # network params
    num_channels: int = 128
    num_layers: int = 6
    resnet_v2: bool = True
    # selfplay params
    selfplay_batch_size: int = 1024
    num_simulations: int = 32
    max_num_steps: int = 256
    # training params
    training_batch_size: int = 4096
    learning_rate: float = 0.001
    # eval params
    eval_interval: int = 5

class _CheckpointUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module in {"__main__", "train"} and name == "Config":
            return _CheckpointConfig
        return super().find_class(module, name)


def load_checkpoint(path):
    with path.open("rb") as file:
        return _CheckpointUnpickler(file).load()


def checkpoint_files(path):
    if path.is_file():
        return [path]
    return sorted(path.glob("*.ckpt"), key=lambda p: (int(p.stem), p.name))


def make_eval_apply(env, config, forward=None):
    if forward is None:
        def forward_fn(x, is_eval=False):
            net = AZNet(
                num_actions=env.num_actions,
                num_channels=config.num_channels,
                num_blocks=config.num_layers,
                resnet_v2=config.resnet_v2,
            )
            return net(x, is_training=not is_eval, test_local_stats=False)

        forward = hk.without_apply_rng(hk.transform_with_state(forward_fn))

    def eval_apply(params, state, observation):
        return forward.apply(params, state, observation, is_eval=True)

    return eval_apply


def make_recurrent_fn(env, eval_apply):
    step = jax.vmap(env.step)

    def recurrent_fn(model, rng_key, action, state):
        del rng_key
        model_params, model_state = model
        current_player = state.current_player
        state = step(state, action)

        (logits, value), _ = eval_apply(
            model_params,
            model_state,
            state.observation,
        )
        logits = logits - jnp.max(logits, axis=-1, keepdims=True)
        logits = jnp.where(
            state.legal_action_mask,
            logits,
            jnp.finfo(logits.dtype).min,
        )

        reward = state.rewards[
            jnp.arange(state.rewards.shape[0]),
            current_player,
        ]
        value = jnp.where(state.terminated, 0.0, value)
        discount = jnp.where(state.terminated, 0.0, -jnp.ones_like(value))

        return mctx.RecurrentFnOutput(
            reward=reward,
            discount=discount,
            prior_logits=logits,
            value=value,
        ), state

    return recurrent_fn


def make_episode_rollout(env, action_fn, max_steps, debug):
    step = jax.vmap(env.step)

    @jax.jit
    def rollout(model, keys, key):
        initial_state = jax.vmap(env.init)(keys)
        batch_size = keys.shape[0]

        def select_by_done(value, reset, done):
            shape = (batch_size,) + (1,) * (value.ndim - 1)
            return jnp.where(done.reshape(shape), reset, value)

        def keep_finished(current, stepped, done):
            shape = (batch_size,) + (1,) * (current.ndim - 1)
            return jnp.where(done.reshape(shape), current, stepped)

        def advance(carry):
            state, done, key = carry
            key, search_key = jax.random.split(key)

            # Finished episodes use a fresh initial state for the extra batched
            # search work, while their terminal state remains in the rollout.
            search_state = jax.tree_util.tree_map(
                lambda value, reset: select_by_done(value, reset, done),
                state,
                initial_state,
            )
            stepped = step(search_state, action_fn(model, search_state, search_key))
            next_state = jax.tree_util.tree_map(
                lambda current, candidate: keep_finished(current, candidate, done),
                state,
                stepped,
            )
            next_done = done | stepped.terminated | stepped.truncated
            return next_state, next_done, key

        def scan_step(carry, step_index):
            state, done, key = jax.lax.cond(
                jnp.all(carry[1]),
                lambda carry: carry,
                advance,
                carry,
            )
            if debug:
                jax.debug.print(
                    "[jax] step {step}/{total}, active episodes: {active}",
                    step=step_index + 1,
                    total=max_steps,
                    active=(~done).sum(),
                    ordered=True,
                )
            return (state, done, key), (state, done)

        state = initial_state
        done = jnp.zeros((batch_size,), dtype=jnp.bool_)
        _, (states, done_history) = jax.lax.scan(
            scan_step,
            (state, done, key),
            jnp.arange(max_steps),
        )
        states = jax.tree_util.tree_map(
            lambda initial, trajectory: jnp.concatenate((initial[None], trajectory), axis=0),
            initial_state,
            states,
        )
        done_history = jnp.concatenate((done[None], done_history), axis=0)
        return states, done_history

    return rollout


def make_rollout(env, eval_apply, recurrent_fn, num_simulations, max_steps, debug):
    def mcts_action(model, state, key):
        (logits, value), _ = eval_apply(
            model[0],
            model[1],
            state.observation,
        )
        root = mctx.RootFnOutput(
            prior_logits=logits,
            value=value,
            embedding=state,
        )
        policy_output = mctx.gumbel_muzero_policy(
            params=model,
            rng_key=key,
            root=root,
            recurrent_fn=recurrent_fn,
            num_simulations=num_simulations,
            invalid_actions=~state.legal_action_mask,
            qtransform=mctx.qtransform_completed_by_mix_value,
            gumbel_scale=1.0,
        )
        return policy_output.action

    return make_episode_rollout(env, mcts_action, max_steps, debug)


def extract_episode(states, done_history, episode):
    terminal_steps = np.flatnonzero(done_history[:, episode])
    if not len(terminal_steps):
        raise RuntimeError(
            f"Episode {episode} did not finish within {done_history.shape[0] - 1} steps"
        )
    frame_count = int(terminal_steps[0]) + 1
    return [
        jax.tree_util.tree_map(lambda value: value[step, episode], states)
        for step in range(frame_count)
    ]


def write_png(state, path):
    path.write_bytes(state_to_png(state))


def state_to_png(state):
    return cairosvg.svg2png(bytestring=state.to_svg().encode("utf-8"))


def _save_gif(frames, target, frame_duration):
    frames[0].save(
        target,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=round(frame_duration * 1000),
        loop=0,
    )


def write_gif(frame_paths, path, frame_duration):
    frames = []
    for frame_path in frame_paths:
        with Image.open(frame_path) as frame:
            frames.append(frame.convert("RGB"))

    _save_gif(frames, path, frame_duration)


def states_to_gif(states, frame_duration=0.1):
    frames = [
        Image.open(io.BytesIO(state_to_png(state))).convert("RGB")
        for state in states
    ]
    output = io.BytesIO()
    _save_gif(frames, output, frame_duration)
    return output.getvalue()


def existing_frames(output_dir, checkpoint_name, episode):
    prefix = f"{checkpoint_name}_{episode}_"
    return sorted(
        (
            path
            for path in output_dir.glob(f"{prefix}*.png")
            if path.stem[len(prefix) :].isdigit()
        ),
        key=lambda path: int(path.stem[len(prefix) :]),
    )


def check_outputs(output_dir, checkpoints, episodes, make_gif, frame_duration):
    collisions = []
    for checkpoint in checkpoints:
        checkpoint_name = checkpoint.stem
        for episode in range(episodes):
            frames = existing_frames(output_dir, checkpoint_name, episode)
            gif = output_dir / f"{checkpoint_name}_{episode}.gif"

            if frames:
                if make_gif and not gif.exists():
                    write_gif(frames, gif, frame_duration)
                collisions.extend(frames)
            elif make_gif and gif.exists():
                collisions.append(gif)

    if collisions:
        names = "\n".join(str(path) for path in collisions)
        raise FileExistsError(f"Render output already exists:\n{names}")


def render_episode(states, output_dir, checkpoint_name, episode, make_gif, frame_duration):
    frames = []
    for step, state in enumerate(states):
        path = output_dir / f"{checkpoint_name}_{episode}_{step}.png"
        write_png(state, path)
        frames.append(path)
        print(f"    wrote frame {step + 1}/{len(states)}: {path.name}")

    if make_gif:
        write_gif(frames, output_dir / f"{checkpoint_name}_{episode}.gif", frame_duration)
        print(f"    wrote gif: {checkpoint_name}_{episode}.gif")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_path", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory; default: renders/<checkpoint-path-name>",
    )
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-simulations", type=int)
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Maximum rollout steps; default: checkpoint config.max_num_steps",
    )
    parser.add_argument("--frame-duration", type=float, default=0.1)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print every JAX rollout step while debugging the render",
    )
    parser.add_argument(
        "--gif",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Create GIFs in addition to PNG frames (default: enabled)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoints = checkpoint_files(args.checkpoint_path)
    output_dir = args.output_dir or Path("renders") / args.checkpoint_path.name
    output_dir.mkdir(parents=True, exist_ok=True)
    check_outputs(
        output_dir,
        checkpoints,
        args.episodes,
        args.gif,
        args.frame_duration,
    )

    for checkpoint_index, checkpoint_path in enumerate(checkpoints):
        checkpoint_start = time.perf_counter()
        print(
            f"[{checkpoint_index + 1}/{len(checkpoints)}] "
            f"Loading {checkpoint_path.name}..."
        )
        checkpoint = load_checkpoint(checkpoint_path)
        config = checkpoint["config"]
        env = pgx.make(config.env_id)
        eval_apply = make_eval_apply(env, config)
        recurrent_fn = make_recurrent_fn(env, eval_apply)
        model = tuple(jax.device_put(value) for value in checkpoint["model"])
        num_simulations = args.num_simulations or config.num_simulations
        max_steps = args.max_steps if args.max_steps is not None else config.max_num_steps
        print(
            f"  env={config.env_id}, episodes={args.episodes}, "
            f"simulations={num_simulations}, max_steps={max_steps}"
        )

        rollout = make_rollout(
            env,
            eval_apply,
            recurrent_fn,
            num_simulations,
            max_steps,
            args.debug,
        )
        rng_key = jax.random.PRNGKey(args.seed + checkpoint_index * args.episodes)
        rollout_key, init_key = jax.random.split(rng_key)
        keys = jax.random.split(init_key, args.episodes)
        print("  Tracing and compiling batched JAX rollout...")
        compile_start = time.perf_counter()
        compiled_rollout = rollout.lower(model, keys, rollout_key).compile()
        print(f"  Compilation complete in {time.perf_counter() - compile_start:.2f}s")
        print("  Running rollout on device...")
        rollout_start = time.perf_counter()
        trajectory, done_history = compiled_rollout(model, keys, rollout_key)
        trajectory, done_history = jax.device_get((trajectory, done_history))
        print(
            f"  Rollout complete in {time.perf_counter() - rollout_start:.2f}s "
            f"({done_history[-1].sum()}/{args.episodes} episodes finished)"
        )

        for episode in range(args.episodes):
            states = extract_episode(trajectory, done_history, episode)
            print(f"  Rendering episode {episode + 1}/{args.episodes} ({len(states)} frames)...")
            render_episode(
                states,
                output_dir,
                checkpoint_path.stem,
                episode,
                args.gif,
                args.frame_duration,
            )
            print(
                f"  Episode {episode + 1}/{args.episodes} complete -> {output_dir}"
            )
        print(
            f"  Checkpoint complete in {time.perf_counter() - checkpoint_start:.2f}s\n"
        )


if __name__ == "__main__":
    main()
