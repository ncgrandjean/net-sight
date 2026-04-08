"""CLI entry point for net-sight."""

import sys
import traceback

from net_sight.pipeline import run


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = [a for a in sys.argv[1:] if a.startswith("--")]

    if len(args) != 1:
        print("Usage: net-sight <image.png> [--debug]")
        sys.exit(1)

    image_path = args[0]
    debug = "--debug" in flags

    try:
        output = run(image_path, debug=debug)
        print(f"\nReport written to: {output}")
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(130)
    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
