$ErrorActionPreference = "Stop"
uv python install 3.11
uv sync --extra dev
uv lock
uv run python -c "import robust_airfoil, torch; print(robust_airfoil.__version__); print(torch.__version__); print('cuda', torch.cuda.is_available())"
