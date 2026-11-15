# Setup

## Python environment

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then from the repo root:

```bash
uv sync --group dev              # Python 3.12, the package, pytest and ruff
uv sync --group dev --extra ml   # also torch, transformers, sentence-transformers
uv run pytest
```

`uv` downloads Python 3.12 if you don't have it, and the same command works on macOS,
Windows and Linux. Run everything through `uv run` (`uv run research ...`,
`uv run python ...`) so it uses the project environment.

## GPU for sentiment (AMD, Windows)

Sentiment scoring runs on CPU by default. On an AMD Radeon RX 6000 card (gfx1030, for
example the RX 6950 XT) it can run on the GPU with AMD's ROCm builds of PyTorch for
Windows. The project's `torch` pin is the CPU build, so use a separate environment rather
than `uv sync --extra ml`, which would put the CPU build back. Make it outside the clone
(the commands below put it next to it) and run them from the repo root in PowerShell:

```powershell
$venv = "..\research_venv_rocm"
uv venv --seed --python 3.12 $venv
uv export --no-hashes --extra ml --group dev --no-emit-project -o "$venv\requirements.txt"
uv pip install --python "$venv\Scripts\python.exe" -r "$venv\requirements.txt"
& "$venv\Scripts\python" -m pip uninstall -y torch
& "$venv\Scripts\python" -m pip install --index-url https://repo.amd.com/rocm/whl-multi-arch/ "torch[device-gfx1030]==2.12.0+rocm7.14.1"
uv pip install --python "$venv\Scripts\python.exe" -e . --no-deps
```

Install the ROCm torch last. The locked requirements pin the CPU `torch`, and even with
that line removed another package's dependency on torch brings the CPU build back over a
ROCm torch installed before them. The only sign is `Torch not compiled with CUDA enabled`
when sentiment asks for the GPU. 2.12 is the newest ROCm build AMD publishes, older than
the project's pin; `python -m pip check` should report no broken requirements.

Only a current Adrenalin driver is needed; the HIP SDK is not. Check with
`& "$venv\Scripts\python" -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"`,
which should print a `+rocm` version and the card: ROCm shows the card as a `cuda`
device, so `--device 0` works unchanged.

`research ingest --sentiment --device 0 --workers 8` then analyzes about 7.7 transcripts a
second on that card with a 12-core CPU (about 18 hours for 507k), after about 45 seconds
of start-up, against about 0.46 on CPU. Sentiment matches the CPU run to within 1e-6 and
every other measure is identical.

Keep CPU-only work in the regular environment: the ROCm build of torch does CPU math about
12 times slower than the CPU build, so `--device -1` in the ROCm environment is very slow.

## Settings

Commands read these environment variables. `.env.example` lists them; uv does not load a
`.env` file by itself, so export them in your shell or pass `uv run --env-file .env`.

| Variable | Needed by | Example |
|---|---|---|
| `DATABASE_URL` | `ingest`, `backfill` (write role); optional for `export` | `postgresql://research_app@<db-host>:5432/research` |
| `TEST_DATABASE_URL` | `pytest -m db` only; a throwaway database, never `research` | `postgresql://postgres@localhost:5432/scratch_test` |

Keep passwords out of URLs. Put them in the PostgreSQL password file instead:
`~/.pgpass` on macOS and Linux (`chmod 600`), `%APPDATA%\postgresql\pgpass.conf` on
Windows, one line per server:

```
<db-host>:5432:research:research_ro:<password>
```

## Database access

The database runs on a group machine reachable over the group's private network. Ask the
maintainer for network access and a role:

- `research_ro`: read only, for analysis and exports. This is what most people need.
- `research_app`: writes rows, for running `ingest` and `backfill`.

Then pull data with a saved or ad-hoc query:

```bash
export DATABASE_URL=postgresql://research_ro@<db-host>:5432/research
uv run research export --sql-file db/queries/events.sql --out results/events.csv.gz
uv run research export --sql "select * from public.keyword_sets" --out results/sets.csv
```

`research export` forces a read-only session and never overwrites a file.

## WRDS

`research returns fetch`, `mcap` and `surprise` pull from WRDS with your own WRDS account.
Create the `~/.pgpass` entry once, for example by running `wrds.Connection()` in Python, or
by adding the line `wrds-pgdata.wharton.upenn.edu:9737:wrds:<wrds username>:<password>`.
WRDS may send a Duo push when a connection opens.
