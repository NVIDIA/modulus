import os
import subprocess

import pytest

# Collecting all the Python files in the scripts directory
script_files = [f for f in os.listdir("test_scripts/") if f.endswith('.py')]

#print(script_files)
@pytest.mark.parametrize("script_file", script_files)
def test_script_execution(script_file):
    """Test if a script runs without error."""
    filepath = os.path.join("test_scripts/", script_file)
    print(filepath)
    result = subprocess.run(["python", filepath], capture_output=True, text=True)

    # Check that the script executed successfully
    assert result.returncode == 0, f"Script {script_file} failed with error:\n{result.stderr}"
