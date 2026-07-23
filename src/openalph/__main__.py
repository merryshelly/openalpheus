import sys

from openalph.cli import main

if __name__ == "__main__":
    # BUG-12: propagate the exit code. main() returns 1 on ConfigError and 130
    # on interrupt, but calling it bare discarded that, so `python -m openalph
    # run ... && next` treated a failed start as success.
    sys.exit(main())
