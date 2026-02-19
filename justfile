# Alerta Escuela Equity Audit

# Install Python dependencies
sync:
    uv sync

# Run all tests
test:
    uv run pytest tests/

# Run gate tests only
gates:
    uv run pytest tests/gates/ -v

# Run unit tests only
unit:
    uv run pytest tests/unit/ -v

# Download ENAHO + admin data
download:
    uv run python src/data/download.py

# Lint code
lint:
    ruff check src/ tests/

# Format code
fmt:
    ruff format src/ tests/

# Start JupyterLab
lab:
    uv run jupyter lab

# Build PDF from LaTeX
pdf:
    cd paper && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex
