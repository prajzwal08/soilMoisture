# `UNetDecoder` — line-by-line walkthrough

Reference: [`model.py`](../model.py) lines **188–270** (class body), plus the call sites at
`model.py:499-510` (`_get_skip_connections`) and `model.py:700-720`
(`SoilMoistureModel.forward`).

---

## What the class is for

`UNetDecoder` takes the transformer's **14×14 spatial bottleneck** and blows it up to a
**224×224 soil-moisture map with `n_depths` channels**, re-injecting TerraMind's shallower
features (L9 / L6 / L3) on the way up, each one FiLM-modulated by the temporal context
vector so the pixel-level decoder "knows" recent weather and history.

---

## Flowchart

```
 INPUTS (all from SoilMoistureModel.forward, model.py:700-720)
 ┌──────────────────────┐ ┌───────────────┐ ┌───────────────┐ ┌───────────────┐ ┌──────────────┐ ┌────────────────────┐
 │ bottleneck           │ │ skip_L9       │ │ skip_L6       │ │ skip_L3       │ │ context      │ │ depth_ctx (opt.)   │
 │ (B,768,14,14)        │ │ (B,768,14,14) │ │ (B,768,14,14) │ │ (B,768,14,14) │ │ (B,768)      │ │ (B,n_depths,768)   │
 │ transformer output   │ │ TerraMind raw │ │ TerraMind raw │ │ TerraMind raw │ │ mean of non- │ │ depth CLS tokens   │
 │ reshaped, L707       │ │ L505-510      │ │               │ │               │ │ spatial toks │ │ L703               │
 └──────────┬───────────┘ └───────┬───────┘ └───────┬───────┘ └───────┬───────┘ └──┬───┬───┬───┘ └─────────┬──────────┘
            │                     │                 │                 │            │   │   │              │
            ▼ L246                ▼ L247            ▼ L248            ▼ L249       │   │   │              │
   ┌─────────────────┐   ┌──────────────┐   ┌──────────────┐  ┌──────────────┐     │   │   │              │
   │ bottle_proj     │   │ skip_proj[0] │   │ skip_proj[1] │  │ skip_proj[2] │     │   │   │              │
   │ 1×1 768→512     │   │ 1×1 768→512  │   │ 1×1 768→256  │  │ 1×1 768→128  │     │   │   │              │
   │ (L211)          │   │ (L213)       │   │ (L214)       │  │ (L215)       │     │   │   │              │
   └────────┬────────┘   └──────┬───────┘   └──────┬───────┘  └──────┬───────┘     │   │   │              │
            │                   ▼                  ▼                 ▼             │   │   │              │
            │            ┌──────────────┐   ┌──────────────┐  ┌──────────────┐     │   │   │              │
            │            │ film_s9      │◄──┤ film_s6      │◄─┤ film_s3      │◄────┘   │   │              │
            │            │ scale*x+shift│   │ (L220)       │  │ (L221)       │◄────────┘   │              │
            │            │ (L219)       │   └──────┬───────┘  └──────┬───────┘◄────────────┘              │
            │            └──────┬───────┘          │                 │                                    │
            │                   │ s9 (B,512,14,14) │ s6              │ s3                                 │
            ▼ x (B,512,14,14)   │                  │                 │                                    │
   ┌─────────────────┐          │                  │                 │                                    │
   │ up1  bilinear×2 │ L251     │                  │                 │                                    │
   │ → (B,512,28,28) │          │                  │                 │                                    │
   └────────┬────────┘          │                  │                 │                                    │
            │      ┌────────────▼───────────┐      │                 │                                    │
            └─────►│ cat( x, interp(s9→28) )│ L252 │                 │                                    │
                   │ → (B,1024,28,28)       │      │                 │                                    │
                   └────────────┬───────────┘      │                 │                                    │
                                ▼                  │                 │                                    │
                   ┌────────────────────────┐      │                 │                                    │
                   │ conv1 _ConvBlock       │ L224 │                 │                                    │
                   │ 1024→256, (B,256,28,28)│      │                 │                                    │
                   └────────────┬───────────┘      │                 │                                    │
                                ▼ L254             │                 │                                    │
                   ┌────────────────────────┐      │                 │                                    │
                   │ up2 → (B,256,56,56)    │      │                 │                                    │
                   └────────────┬───────────┘      │                 │                                    │
                   ┌────────────▼───────────┐      │                 │                                    │
                   │ cat( x, interp(s6→56) )│◄─────┘  L255           │                                    │
                   │ → (B,512,56,56)        │                        │                                    │
                   └────────────┬───────────┘                        │                                    │
                                ▼                                    │                                    │
                   ┌────────────────────────┐ L227                   │                                    │
                   │ conv2 512→128 (56×56)  │                        │                                    │
                   └────────────┬───────────┘                        │                                    │
                                ▼ L257                               │                                    │
                   ┌────────────────────────┐                        │                                    │
                   │ up3 → (B,128,112,112)  │                        │                                    │
                   └────────────┬───────────┘                        │                                    │
                   ┌────────────▼───────────┐                        │                                    │
                   │ cat( x, interp(s3→112))│◄───────────────────────┘  L258                              │
                   │ → (B,256,112,112)      │                                                             │
                   └────────────┬───────────┘                                                             │
                                ▼                                                                         │
                   ┌────────────────────────┐ L230                                                        │
                   │ conv3 256→64 (112×112) │                                                             │
                   └────────────┬───────────┘                                                             │
                                ▼ L260  (no skip left — L12 grid exhausted)                               │
                   ┌────────────────────────┐                                                             │
                   │ up4 → (B,64,224,224)   │                                                             │
                   └────────────┬───────────┘                                                             │
                                ▼ L261                                                                    │
                   ┌────────────────────────┐                                                             │
                   │ conv4 64→64            │                                                             │
                   └────────────┬───────────┘                                                             │
                                ▼ L262                                                                    │
                   ┌────────────────────────┐                                                             │
                   │ pre_head_drop  p=0.1   │                                                             │
                   └────────────┬───────────┘                                                             │
                                ▼                                                                         │
                   ┌────────────────────────────────────────┐                                             │
                   │  use_cls_depth AND depth_ctx not None? │  L264                                       │
                   └───────┬──────────────────────┬─────────┘                                             │
                      NO   │                      │  YES                                                  │
                           ▼                      ▼  loop d = 0..n_depths-1 (L266)                         │
              ┌────────────────────┐   ┌──────────────────────────────┐                                   │
              │ head: 1×1 64→3     │   │ depth_film[d](x, ────────────────────────────────────────────────┘
              │ (L241, L270)       │   │            depth_ctx[:,d,:]) │ L267
              │ ONE shared feature │   │  then heads[d]: 1×1 64→1     │ L268
              │ map, 3 biased      │   └──────────────┬───────────────┘
              │ slices             │                  ▼ cat over dim=1 (L269)
              └─────────┬──────────┘   ┌────────────────────────────┐
                        │              │ per-depth feature maps     │
                        └──────────────┴────────────┬───────────────┘
                                                    ▼
                                    OUTPUT  (B, n_depths, 224, 224)
```

---

## Line by line — `__init__` (L197–241)

| Line | What |
|---|---|
| L199-204 | `in_ch=768` (TerraMind / transformer width), `skip_ch=768`, `dec_ch=(512,256,128,64)` — channels shrink as resolution grows; `n_depths=3` output depth bins; `d_context=768` FiLM conditioning width |
| L211 | `bottle_proj` — 1×1 conv, pure channel squeeze 768→512, no spatial mixing |
| L212-216 | Three 1×1 convs projecting each skip to the channel count of the stage it will be concatenated into (512 / 256 / 128) |
| L219-221 | One `FiLMLayer` per skip. `FiLMLayer` (L150-168) maps `context (B,768)` → `(B,2C)`, splits into per-channel scale/shift, applies `scale*x + shift`. Weights are zero-init and bias is (1,0)-init (L158-160) so at step 0 it's the **identity** — training starts from a plain U-Net and *learns* to use weather context |
| L223-233 | Four `Upsample(×2, bilinear)` + `_ConvBlock` pairs. `_ConvBlock` (L171-185) = Conv3×3 → BN → ReLU → Conv3×3 → BN → ReLU → `Dropout2d(0.15)`. The `c[i]+c[i]` in-channels (L224/227/230) is the concat of upsampled `x` and the skip. `conv4` (L233) has no skip so it's `c[3]→c[3]` |
| L235 | `Dropout(0.1)` right before the head — element-wise (not channel-wise like `Dropout2d`) |
| L237-241 | Two head modes. **Default** (`else`, L241): a single 1×1 conv 64→`n_depths` — all depths share the same 64-d feature map and differ only by `64×3+3 = 195` parameters. **`use_cls_depth=True`** (L238-239): each depth gets its own FiLM *and* its own 1×1 head, so each depth can re-weight the shared features before predicting |

## Line by line — `forward` (L243–270)

| Line | What |
|---|---|
| L246 | Bottleneck 768→512, still 14×14 |
| L247-249 | Each skip: 1×1 project → FiLM with the global temporal `context`. The skips are the **raw** TerraMind L3/L6/L9 of the anchor acquisition (`_get_skip_connections`, L505-510) — they never pass through the transformer, so the FiLM is what makes them time-aware |
| L251-252 | 14→28, concat FiLM'd **L9** (deepest skip → coarsest stage), conv → 256 ch |
| L254-255 | 28→56, concat **L6**, conv → 128 ch |
| L257-258 | 56→112, concat **L3** (shallowest skip → finest stage), conv → 64 ch |
| L260-261 | 112→224, conv only (no skip left) |
| L262 | Dropout |
| L264-269 | Per-depth branch: FiLM the shared 224×224 feature map with that depth's CLS vector, apply that depth's 1×1 head, concat the `n_depths` single-channel maps |
| L270 | Shared branch: one 1×1 conv emits all depths at once |

---

## Two things worth noting from the code

### 1. The `F.interpolate` calls at L252 / L255 / L258 are doing real work, not shape-safety

All three skips are natively 14×14 — `_get_skip_connections` (L505-510) reshapes 196 tokens
into a 14×14 grid — so they get **bilinearly blown up** to 28 / 56 / 112. A classic U-Net
gets genuinely higher-resolution skips from the encoder; here every source of spatial
information (bottleneck *and* all three skips) lives on the same 14×14 grid. Everything
finer than ~16 px per cell is bilinear interpolation plus learned 3×3 convs.

That is the decoder-side smoothness identified in the Tier-1 verdict — the fix is dense
spatial supervision, not more attention.

### 2. `use_cls_depth=False` gives only **195** depth-specific parameters

Out of the entire decoder, the default head (L241) contributes `64×3 + 3 = 195` parameters
that can distinguish one depth from another — every depth reads the same 64-channel feature
map. This is consistent with the surface depth fitting while 30–100 cm regresses.
`use_cls_depth=True` (L237-239) adds a full FiLM (`768 → 2×64` = 98 k params per depth) plus
a separate 1×1 head per depth, letting each depth re-weight the shared features first.
