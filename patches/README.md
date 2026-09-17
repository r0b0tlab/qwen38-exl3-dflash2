The `dflash2-pathway` format-patch series is historical.

Upstream ExLlamaV3 landed a native DFlash2 implementation (CUDA `dflash2_dynconv`,
`dflash2_topk`, `dflash2_selector_walk`) that superseded this overlay. The published
runtime now tracks `r0b0tlab/exllamav3` branch `community` (native DFlash2 + GDN
rewind + dense M=32/64 TILEBLOCKS_M). Do not apply these patches onto current
engine trees.
