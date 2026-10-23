#!/usr/bin/env bash
# Download the URL list and unpack SuiteSparse Matrix Market archives.
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [[ ${1:-} == --help || $# -gt 2 ]]; then
    echo "Usage: $0 [URL_LIST [OUTPUT_DIRECTORY]]"
    echo "Defaults: $root/matrices.txt and $root/matrices"
    exit 0
fi
list=${1:-"$root/matrices.txt"}
destination=${2:-"$root/matrices"}
[[ -f "$list" ]] || { echo "URL list not found: $list" >&2; exit 1; }
for tool in curl tar; do
    command -v "$tool" >/dev/null || { echo "Required command missing: $tool" >&2; exit 1; }
done
mkdir -p "$destination"

while IFS= read -r url || [[ -n "$url" ]]; do
    url=${url%$'\r'}
    [[ -z "$url" || "$url" == \#* ]] && continue
    filename=${url##*/}
    case "$filename" in
        *.tar.gz|*.mtx) ;;
        *) echo "Unsupported URL (expected .mtx or .tar.gz): $url" >&2; exit 1 ;;
    esac
    target="$destination/$filename"
    if [[ -f "$target" ]]; then
        echo "Already downloaded: $filename"
    else
        echo "Downloading: $filename"
        # Only completed downloads get their final names. A failed transfer
        # leaves .part in place so the next invocation can resume it.
        curl --fail --location --show-error --retry 3 --connect-timeout 30 \
            --continue-at - --output "$target.part" "$url"
        mv "$target.part" "$target"
    fi

    if [[ "$filename" == *.tar.gz ]]; then
        name=${filename%.tar.gz}
        marker="$target.extracted"
        matrix="$destination/$name/$name.mtx"
        if [[ -f "$marker" && -f "$matrix" && ! "$target" -nt "$marker" ]]; then
            echo "Already extracted: $name"
        else
            echo "Extracting: $filename"
            tar -xzf "$target" -C "$destination"
            [[ -f "$matrix" ]] || {
                echo "Expected main matrix not found: $matrix" >&2
                exit 1
            }
            touch "$marker"
        fi
    else
        matrix="$target"
    fi
    IFS= read -r header < "$matrix" || true
    [[ "$header" == '%%MatrixMarket '* ]] || {
        echo "Not a Matrix Market file: $matrix" >&2
        exit 1
    }
    echo "Ready: $matrix"
done < "$list"

echo "Finished. Matrices are in: $destination"
