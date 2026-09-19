"""Competition entry point: write <id>.png to a separate output directory."""

import sys

from run_wavelet import main


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
