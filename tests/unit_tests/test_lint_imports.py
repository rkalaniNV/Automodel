import pytest
import subprocess
import sys

def test_import_linter():
    """Test that import graph contracts are satisfied"""
    # Install import-linter if not available
    try:
        import importlinter  # noqa
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--break-system-packages", "import-linter"])

    # Run the lint-imports command
    result = subprocess.run(
        ["lint-imports", "--debug", "--verbose", "--no-cache"],
        cwd=".",  # Run from project root
        capture_output=True,
        text=True
    )

    assert result.returncode == 0, (
        f"Import linting failed (exit code {result.returncode}):\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )
