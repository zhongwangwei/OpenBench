"""Run all manual generators in sequence."""
from __future__ import annotations

import sys

from . import (
    generate_config_schema,
    generate_internal_interfaces,
    generate_model_table,
    generate_reference_table,
    generate_registry_schema,
)


GENERATORS = [
    ("reference_table", generate_reference_table.main),
    ("model_table", generate_model_table.main),
    ("config_schema", generate_config_schema.main),
    ("registry_schema", generate_registry_schema.main),
    ("internal_interfaces", generate_internal_interfaces.main),
]


def main() -> int:
    failures: list[str] = []
    for name, fn in GENERATORS:
        print(f"\n=== {name} ===", file=sys.stderr)
        try:
            fn()
        except SystemExit as e:
            if e.code not in (0, None):
                failures.append(f"{name}: exit {e.code}")
        except Exception as e:
            failures.append(f"{name}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
    if failures:
        print("\nFailures:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print("\nAll generators succeeded.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
