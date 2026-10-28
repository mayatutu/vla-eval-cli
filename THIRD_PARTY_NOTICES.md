# Third-party sources and licenses

This repository assembles locally adapted evaluation code and a limited subset of runtime components. It does not grant a new blanket license to all included files or redistribute model weights.

| Component | Origin | Included material |
|---|---|---|
| Colosseum client | https://github.com/frodobots-org/colosseum-client | Selected robot, recording, protocol and data-export modules with local adaptations |
| DROID | https://github.com/droid-dataset/droid | MiniPC runtime subset with UVC and environment configuration adaptations |
| G05 | https://github.com/OpenGalaxea/GalaxeaVLA | Modified inferencer overlay; upstream notice in third_party/galaxeavla/NOTICE |
| LingBot VLA v2 | Existing deployment source; exact upstream revision unresolved | Backend integration; captured license in third_party/lingbot-vla-v2/LICENSE |
| LAP / OpenPI | https://github.com/lihzha/lap | Backend integration; captured LAP license in third_party/lap/LICENSE |
| Isaac-GR00T | https://github.com/NVIDIA/Isaac-GR00T | Backend integration; captured license in third_party/isaac-gr00t/LICENSE |
| LeRobot | https://github.com/huggingface/lerobot | External model dependency, not vendored as a full package |

The G05 notice identifies its Community License (Non-Commercial + Limited Patent License). The full LICENSE-G0.5 was not present in the inspected snapshot; obtain and follow the applicable terms from the upstream source. Captured licenses do not change model-weight terms.

The inspected local Colosseum/DROID copies did not include recoverable upstream license metadata. Their redistribution terms and exact source revision remain unresolved in this private preparation. Confirm the original terms and retain required notices before public redistribution. This repository is not represented as a fully audited open-source distribution.
