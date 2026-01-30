#!/usr/bin/env python3
import sys
from pathlib import Path

def main():
    if len(sys.argv) != 4:
        print("Usage: ptx_to_header.py <in.ptx> <out.h> <symbol>")
        sys.exit(1)

    ptx_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    symbol   = sys.argv[3]

    ptx_text = ptx_path.read_text()

    delim = "OPTIX_PTX_EMBED"
    out = []
    out.append("#pragma once\n")
    out.append("#include <cstddef>\n\n")
    out.append(f"static const char {symbol}[] = R\"{delim}(\n")
    out.append(ptx_text)
    if not ptx_text.endswith("\n"):
        out.append("\n")
    out.append(f"){delim}\";\n\n")
    out.append(f"static const std::size_t {symbol}_len = sizeof({symbol});\n")

    out_path.write_text("".join(out))

if __name__ == "__main__":
    main()
