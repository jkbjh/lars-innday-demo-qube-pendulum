# inday-qube-demo

Demo for the Quanser-Qube Furuto (rotary)-Pendulum: learning a
PPO+Colored Noise qube swingup agent.


## Setup

- `git clone --recurse-submodules https://github.com/jkbjh/lars-innday-demo-qube-pendulum.git`
  (already cloned: `git submodule update --init --recursive`)
- Prerequisites: `uv`; `pre-commit` and `ruff` installed globally:
  - `uv tool install pre-commit && uv tool install ruff`
  - or `pipx install pre-commit && pipx install ruff`
- Python 3.10 (3.13 untested)

## Install

From the repo root (`deps/quanser-qube` is a relative path):

```
uv pip install -r requirements.built.cpu -e .   # or .cuda; plain pip works the same
```

- `-r requirements.built.cpu`: all (sub-)dependencies, pinned
- `-e .`: this repo itself (`inday_qube_demo`), editable

The built requirements contain the full sources (`cnppo` git URL, `gymqube` local path),
so they work with uv and pip. `pyproject.toml` alone does not work with pip:
its `[tool.uv.sources]` is uv-only, so pip would look both up on PyPI.

### Hardware

Needs the Quanser HIL SDK (`/opt/quanser/hil_sdk` or `/usr/include/quanser`), see [Container](#container).

If gymqube was first built without the SDK (e.g. simulator only), force a rebuild;
uv does not notice changed environment variables or a newly installed SDK:

```
uv pip install --reinstall-package gymqube --refresh-package gymqube -r requirements.built.cpu
```

(pip: `pip install --force-reinstall --no-cache-dir --no-deps ./deps/quanser-qube`)

### Simulator only

No SDK needed: `GYMQUBE_SIMULATOR_ONLY=1 uv pip install -r requirements.built.cpu`

- prints a warning banner during the build and a warning on `import gymqube`
- hardware is unavailable in this mode

## Run

`python -m inday_qube_demo.train`

Flags: `--n-envs`, `--total-timesteps`, `--eval-freq`, `--noise-color`, `--render`, `--video-folder`.

After each evaluation, one deterministic episode is rendered (`--render`):
- `video` (default): `videos/t<step>.mp4` (zero-padded step), works headless (EGL)
- `human`: live window
- `both`: video and live window
- `none`: nothing

## Project layout

- `inday_qube_demo/`: the inday-demo code (all demo parts go here)
- `deps/quanser-qube`: gymqube submodule (branch `inday-demo`)
- `deps/singularity-container`: container definitions for the HIL SDK
- `requirements.{cpu,cuda}`, `Makefile`: torch/jax variants and pip-compile

## Development

```
uv pip install -r requirements.built.cpu -e .   # checked-in state, incl. dev deps (pdbpp, ipython)
pre-commit install  # make repository pre-commit hooks ready.
```

Install with optinally using pinned sub-dependencies:

|           | with pinned sub-dependencies                         | regular install                    |
|-----------|------------------------------------------------------|------------------------------------|
| command   | `uv pip install -r requirements.built.cpu -e .`      | `uv pip install -e ".[dev]"`       |
| versions  | exact pins (reproducible)                            | newest allowed by `pyproject.toml` |
| torch/jax | cpu or cuda variant, as in `requirements.{cpu,cuda}` | PyPI default (CUDA build on Linux) |

`requirements.built.*` are generated and checked in. Rebuild if pyproject.toml or [dev] requirements changed:

```
UV_PYTHON=3.10 make all     # keeps existing pins
make -n -B                  # only print the compile commands
```

Update or downgrade a single package: run the printed command with `--upgrade-package`, e.g.

```
uv pip compile --emit-index-url --extra dev requirements.cpu pyproject.toml \
  --output-file=requirements.built.cpu --upgrade-package torch          # upgrade
  # --upgrade-package numpy==1.26.4                                     # pin/downgrade
```

## Container

Build and run instructions: [deps/singularity-container/README.md](deps/singularity-container/README.md)

```
sudo singularity build quanser-deb.sif quanser-py.def
singularity -v exec --home $(pwd) quanser-deb.sif /bin/bash --noprofile
```

## Known issues

- the submodule must be on the `inday-demo` branch
- Python 3.13 and the cuda requirements are untested

## References

- Colored Noise in PPO: https://arxiv.org/abs/2312.11091
- cnppo code: https://github.com/jkbjh/cn-ppo-paper-code
