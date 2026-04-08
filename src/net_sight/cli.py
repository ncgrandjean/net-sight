"""CLI entry point for net-sight."""

import sys
import traceback

from net_sight.pipeline import run


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a.split("=")[0]: a.split("=")[1] if "=" in a else True for a in sys.argv[1:] if a.startswith("--")}

    if len(args) != 1:
        print("Usage: net-sight <image.png> [--debug] [--from=N]")
        print()
        print("Options:")
        print("  --debug     Save intermediate images to debug/ folder")
        print("  --from=N    Resume VLM analysis from tile N (1-based)")
        sys.exit(1)

    image_path = args[0]
    debug = "--debug" in flags
    from_tile = int(flags.get("--from", 0))

    try:
        output = run(image_path, debug=debug, from_tile=from_tile)
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
