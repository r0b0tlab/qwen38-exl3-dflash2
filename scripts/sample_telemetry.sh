#!/usr/bin/env bash
# 2-second telemetry sampler for campaign runs (PROCEDURES.md §4).
# Writes TSV: epoch_s, power_w, temp_c, util_pct, gfx_clock_mhz, throttle, vram_mib, mem_available_kb, swap_total_kb, swap_free_kb
# Usage: scripts/sample_telemetry.sh <output.tsv> [interval_seconds]
set -u
OUT="$1"
INTERVAL="${2:-2}"
printf 'epoch_s\tpower_w\ttemp_c\tutil_pct\tgfx_clock_mhz\tthrottle\tvram_mib\tmem_available_kb\tswap_total_kb\tswap_free_kb\n' > "$OUT"
while true; do
    NOW=$(date +%s)
    read -r PWR TMP UTL CLK THR VRM <<< "$(nvidia-smi --query-gpu=power.draw,temperature.gpu,utilization.gpu,clocks.current.graphics,clocks_throttle_reasons.active,memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | tr ',' ' ')"
    read -r MAV ST SF <<< "$(awk '/MemAvailable/{a=$2} /SwapTotal/{t=$2} /SwapFree/{f=$2} END{print a" "t" "f}' /proc/meminfo)"
    if [ -n "${PWR:-}" ]; then
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$NOW" "$PWR" "$TMP" "$UTL" "$CLK" "$THR" "$VRM" "$MAV" "$ST" "$SF" >> "$OUT"
    fi
    sleep "$INTERVAL"
done
