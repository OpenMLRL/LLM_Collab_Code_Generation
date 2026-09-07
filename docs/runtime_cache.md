# Slurm CUDA cache

Training entrypoints configure a private, node-local CUDA JIT cache at the start
of `main()`, before loading models or initializing CUDA. This prevents independent
jobs from waiting on the same `~/.nv/ComputeCache/index` on shared NFS storage.
The existing training commands do not need to change.

Update **CoMLRL together with this repository**: the launch hook requires
`comlrl.runtime.configure_job_cuda_cache` on the corresponding `main`/`cen` branch.
No trainer algorithm, reward, sampling count, seed or training budget is changed.

- Non-Slurm launches are unchanged.
- Explicit `CUDA_CACHE_PATH` and `CUDA_CACHE_DISABLE=1` settings are preserved.
- Otherwise an existing local `SLURM_TMPDIR` is preferred, with `/tmp` as the
  fallback. `COMLRL_CUDA_CACHE_ROOT` can specify another existing node-local root.
- A single `[runtime]` stderr line records the path; subprocesses inherit it.
- Shared caches are never deleted. Temporary cache cleanup follows cluster policy.
- Existing running jobs and old frozen snapshots are unchanged by `git pull`.
  Future frozen snapshots must include the updated CoMLRL helper as well.

See [CoMLRL's runtime cache guide](https://github.com/OpenMLRL/CoMLRL/blob/main/docs/runtime_cache.md).

CPU-only entrypoint regression test:

```bash
python -m unittest discover -s tests -p test_runtime_cache_entrypoints.py -v
```
