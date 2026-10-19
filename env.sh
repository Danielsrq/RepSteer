# Source this before running things manually:  source env.sh
# Activates the venv. Models use the standard HF cache (~/.cache/huggingface);
# the project-local `models/` symlink points there for convenience.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/.venv/bin/activate"
echo "venv active | HF cache: ${HF_HOME:-$HOME/.cache/huggingface}"
