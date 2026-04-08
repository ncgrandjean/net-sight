"""CLI entry point for net-sight."""

import sys

from net_sight.pipeline import run


def main():
    if len(sys.argv) != 2:
        print("Usage: net-sight <image.png>")
        sys.exit(1)

    image_path = sys.argv[1]
    try:
        output = run(image_path)
        print(f"\nReport written to: {output}")
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(130)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
