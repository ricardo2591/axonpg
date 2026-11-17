"""AxonPG cross-platform CLI (`axonpg init`, `axonpg doctor`, `axonpg serve`)."""

import argparse


def main():
    parser = argparse.ArgumentParser(prog="axonpg", description="AxonPG: Postgres-native cognitive memory for AI agents.")
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("init", help="Interactive setup wizard (Hermes Agent / OpenClaw MCP / --lite)")
    sub.add_parser("doctor", help="Check Postgres, pgvector, embeddings, and MCP connectivity")
    sub.add_parser("serve", help="Start AxonPG MCP server")
    args = parser.parse_args()
    if not args.cmd:
        parser.print_help()
        return
    print(f"[AxonPG v0.1.0] Command '{args.cmd}' scaffold ready for Sprint 1 implementation.")


if __name__ == "__main__":
    main()
