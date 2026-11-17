"""Module entrypoint allowing `python -m axonpg ...`."""

from axonpg.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
