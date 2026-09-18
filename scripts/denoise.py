"""Competition entry point: write <id>.png to a separate output directory."""

import sys

from run_nafnet import parser, run


if __name__ == "__main__":
    cli = parser(default_suffix="", default_output_dir=None)
    args = cli.parse_args()
    if args.output_dir is None:
        cli.error("--denoised_dir / --output_dir is required for competition outputs.")
    if args.source.resolve() == args.output_dir.resolve():
        cli.error("Use a separate output directory to keep the noisy input folder unchanged.")
    try:
        raise SystemExit(run(args))
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
