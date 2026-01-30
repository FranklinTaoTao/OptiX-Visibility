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

    data = ptx_path.read_bytes()

    # Emit as a byte array to avoid MSVC "string too big" (C2026)
    # Keep formatting friendly: 12 bytes per line
    bytes_per_line = 12

    out = []
    out.append("#pragma once\n")
    out.append("#include <cstddef>\n")
    out.append("#include <cstdint>\n\n")

    out.append(f"// Embedded PTX from: {ptx_path.name}\n")
    out.append(f"static const std::uint8_t {symbol}[] = {{\n")

    for i in range(0, len(data), bytes_per_line):
        chunk = data[i:i+bytes_per_line]
        out.append("  " + ", ".join(f"0x{b:02x}" for b in chunk) + ",\n")

    out.append("};\n\n")
    out.append(f"static const std::size_t {symbol}_len = sizeof({symbol});\n")
    out.append(f"static inline const char* {symbol}_cstr() {{\n")
    out.append(f"  return reinterpret_cast<const char*>({symbol});\n")
    out.append("}\n")

    out_path.write_text("".join(out), encoding="utf-8")

if __name__ == "__main__":
    main()
