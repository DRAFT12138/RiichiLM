# Interface illustrations

The character portraits and salon artwork in this directory are original
AI-generated assets bundled locally with RiichiLM. They do not affect policies,
legal actions, scores, or hidden information.

`salon-interlude.webp` was generated on 2026-11-03 using OpenAI image generation
and is used for loading and table scenery. On 2026-11-04 it was edited with
image generation to remove the unrelated hostess, retaining scenery only.

## Character-specific result illustrations

On 2026-11-04, `victory-0.webp` through `victory-12.webp` were generated with
OpenAI image generation, using the existing library images below as explicit
visual references. Each preserves the selected character's hair, eye color,
accessories, and costume. IDs match `characterNames` in `web/src/cast.ts`.

| ID | Character | Reference |
| --- | --- | --- |
| 0 | 小樱 | `seat-portraits.png`, first column |
| 1 | 小凛 | `seat-portraits.png`, second column |
| 2 | 小澪 | `seat-portraits.png`, third column |
| 3 | 小桃 | `seat-portraits.png`, fourth column |
| 4 | 绯花 | `formal-portraits.webp`, first column |
| 5 | 紫苑 | `formal-portraits.webp`, second column |
| 6 | 青岚 | `formal-portraits.webp`, third column |
| 7 | 花音 | `festival-portraits.webp`, first column |
| 8 | 星乃 | `festival-portraits.webp`, second column |
| 9 | 若叶 | `festival-portraits.webp`, third column |
| 10 | 青羽 | `teal-host.webp` |
| 11 | 桃香 | `school-host.webp` |
| 12 | 小葵 | `cardigan-host.webp` |

Results select the actual winning seat's character. Multiple winners can be
selected individually; final standings show the first-place character. Ordinary
draws show scenery. The unrelated generic `victory-host.webp` has been removed.
Generated PNG originals were encoded as WebP for the browser bundle.

These are original designs, not artwork extracted from Mahjong Soul.
No external image or sound service is contacted at runtime. Sound cues are
original short oscillator sequences synthesized by the browser Web Audio API
in `web/src/sound.tsx`; there are no third-party music or voice recordings.
