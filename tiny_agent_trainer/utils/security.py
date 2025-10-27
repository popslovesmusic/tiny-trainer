import subprocess
import tempfile
import os
import signal
import time
import re
from typing import Dict, Any, Optional, List, Set
from pathlib import Path
import logging
from contextlib import contextmanager
import shutil
import sys  # <-- Import sys to check the OS

logger = logging.getLogger(__name__)

# --- OS-SPECIFIC RESOURCE LIMITS ---

def _set_resource_limits(memory_limit_mb, time_limit_s):
    """Sets memory and CPU time limits for the current process."""
    # This function will only be fully active on non-Windows systems.
    if sys.platform != "win32":
        import resource  # <-- Import is now safely inside the OS check
        
        # Set memory limit (address space)
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        mem_bytes = memory_limit_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, hard))

        # Set CPU time limit
        soft, hard = resource.getrlimit(resource.RLIMIT_CPU)
        resource.setrlimit(resource.RLIMIT_CPU, (time_limit_s, hard))
    else:
        # On Windows, resource limits are not supported this way.
        # We can print a warning or handle it silently.
        logger.warning("Resource limits (memory, CPU time) are not enforced on Windows.")

# --- MAIN FUNCTIONS ---

@contextmanager
def secure_execution(memory_limit_mb=1024, time_limit_s=60):
    """A context manager to execute code within resource constraints."""
    try:
        _set_resource_limits(memory_limit_mb, time_limit_s)
        yield
    except MemoryError:
        logger.error("Execution failed: Memory limit exceeded.")
        raise
    except Exception as e:
        if "CPU time limit exceeded" in str(e):
            logger.error("Execution failed: Time limit exceeded.")
        else:
            logger.error(f"An unexpected error occurred during secure execution: {e}")
        raise

def execute_vsl_safely(vsl_code: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Executes VSL (Julia) code in a sandboxed environment.
    This is a placeholder for a more robust sandboxing implementation.
    """
    if not shutil.which("julia"):
        raise RuntimeError("Julia runtime not found. Please install Julia and ensure it's in your PATH.")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        script_path = tmp_path / "run_vsl.jl"
        data_path = tmp_path / "data.json"
        output_path = tmp_path / "output.json"

        # Write VSL code and data to files
        script_path.write_text(vsl_code)
        with open(data_path, 'w') as f:
            import json
            json.dump(data, f)

        # Execute the Julia script
        command = [
            "julia", str(script_path), str(data_path), str(output_path)
        ]
        
        try:
            with secure_execution():
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=60, # A failsafe timeout
                    check=True # Raise exception for non-zero exit codes
                )

            # Read the output from the file
            with open(output_path, 'r') as f:
                import json
                output_data = json.load(f)
            
            return output_data

        except subprocess.TimeoutExpired:
            logger.error("VSL execution timed out.")
            raise
        except subprocess.CalledProcessError as e:
            logger.error(f"VSL script failed with error:\n{e.stderr}")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred executing VSL: {e}")
            raise