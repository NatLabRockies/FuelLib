"""Format all Python source code using Black."""

import subprocess
import sys


def main():
    """Run Black formatter on all Python files in the repository."""
    try:
        result = subprocess.run(
            ['find', '.', '-name', '*.py', '-print0'],
            capture_output=True,
            text=False,
            check=True,
        )
        
        # Use xargs to pass files to black
        process = subprocess.Popen(
            ['xargs', '-0', 'black'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout, stderr = process.communicate(input=result.stdout)
        
        if stdout:
            print(stdout.decode())
        if stderr:
            print(stderr.decode(), file=sys.stderr)
        
        sys.exit(process.returncode)
    except Exception as e:
        print(f"Error running black formatter: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
