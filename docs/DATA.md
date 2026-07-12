# Data boundary

VitalsSight uses public or provider-controlled camera-based physiology datasets. Raw videos, facial frames, physiological reference files and provider archives are not redistributed in this repository.

The manuscript analyses use protocol-specific roles rather than treating all datasets as one pooled benchmark:

| Dataset/domain | Manuscript role | Public-release boundary |
|---|---|---|
| UBFC-rPPG | Primary internal selector analysis | Obtain from the original provider; preserve participant-disjoint splits |
| MCD-rPPG | Post-exercise and elevated-HR stress | Obtain under provider terms; identifiable frames are not redistributed here |
| UBFC-Phys | Source-shift boundary | Obtain from the original provider; conflicting protocol variants remain separate |
| rPPG-10 | Descriptive external ROI analysis | Obtain from the original provider |
| SCAMPS | Synthetic sanity and failure boundary | Obtain under the original distribution terms |
| MR-NIRP | Low-light/RGB-NIR case audit | Use the provider-released sequences and synchronized reference data under their terms |

Set `CONTACTLESS_DATA_ROOT` or `ADULT_DATA_ROOT` to a local, authorized dataset location. Dataset adapters must retain participant identifiers so that repeated windows remain nested within participants during analysis.

Do not place raw data, participant frames or provider archives inside the Git repository.

