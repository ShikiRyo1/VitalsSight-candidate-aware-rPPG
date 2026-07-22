# Native macOS reviewer delivery

The architecture-specific packages produced by `build-macos-native-app.yml` are the reviewer-facing macOS delivery. They contain a self-contained `VitalsSight.app`; the recipient does not install Python, create a virtual environment, or download Python packages.

## Choose the package

- `AppleSilicon`: Apple M1, M2, M3, M4, M5 and later Apple-silicon Macs.
- `Intel`: older Macs whose About This Mac panel identifies an Intel processor.

Extract the complete ZIP. On the first launch, Control-click `VitalsSight.app`, choose **Open**, and confirm **Open**. This per-application Gatekeeper override is required because the research build is ad-hoc signed rather than Apple-notarized. Do not disable Gatekeeper globally.

The launcher starts the packaged UI and API on `127.0.0.1`, opens the browser, and shows whether the optional local-language model is available. Deterministic assessment, evidence reports and bounded guidance remain available without Ollama. Application state is stored in `~/Library/Application Support/VitalsSightResearchDemo`; logs are stored in `~/Library/Logs/VitalsSight`.

Each delivery is built and tested on the matching GitHub-hosted macOS architecture. The workflow verifies the Mach-O architecture and code signature, starts the packaged UI and API, queries the assistant health and cases endpoints, checks the Streamlit page, validates the output JSON, and tests the final ZIP archive before upload.

The native build uses one pinned OpenCV contrib distribution. MediaPipe requires that distribution, and it includes the standard OpenCV API; installing both standard and contrib wheels would place two owners over the same `cv2` package and is rejected by this packaging contract.

A seamless normal double-click without the first-launch security prompt requires an Apple Developer ID certificate and Apple notarization. Those credentials are not embedded in the repository or research package.
