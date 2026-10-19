# Activation-steering results — self-contained demo

Open **`RESULTS.html`** in any browser (double-click, or drag it into a browser tab).
Everything is offline: no server, no internet, no build step. Each page has its CSS,
JavaScript, and data embedded inline, so this folder is complete on its own — zip it
and it still works anywhere.

## Contents

```
RESULTS.html                        landing page — start here
concepts/concept_tokens.html        I   · grand-mean DiM over emotion vectors
jlens/size_compare.html             II  · J-space occupancy: layer sweep & vs model size
emoji/emoji_steering_story.html     III-1 · emoji: DiM steering attempts + LoReFT A/B
  emoji/compare_all.html              (tab 1, embedded)
  emoji/ab_sysprompt.html             (tab 2, embedded)
reft/steer_compare.html             III-2 · LoReFT vs DiM steering (happy / sad)
security/v2_reader_view.html        IV  · 5-label security readers + J-space wordcloud (~72 MB)
```

## Notes

- `security/v2_reader_view.html` is ~72 MB (all token activations are baked in) — give it
  a few seconds to load. It is the bulk of the folder's size.
- `emoji_steering_story.html` shows its two sub-pages in iframes. Opening from the local
  filesystem (`file://`) renders them fine in Firefox, Chrome, and Safari; the tab toggle
  only shows/hides panes and does not read across frames, so no server is needed.
- Best viewed in a current desktop browser. Pages are theme-aware (light/dark).
