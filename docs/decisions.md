# Decision log

Short, dated entries, newest first. Record what was decided and why.

## 2026-10-15: Project scaffolding (M0)

- **Standards:** Copy the setup from `cost-sensitive-targeting`: Python 3.12, uv, src layout, ruff (numpy docstrings), strict mypy, pytest with coverage, pre-commit with nbstripout, GitHub Actions CI.
- **Dependencies:** Start with numpy, pandas, pyarrow and pyyaml only. Add model and survival libraries in the milestone that first needs them.
- **Data:** Keep all Freddie Mac and FRED files under `data/`, which is gitignored.
