# VitalsSight on macOS

## Recommended reviewer package

Use the architecture-specific native package produced by the `build-macos-native-app` workflow. It contains `VitalsSight.app` and a bundled Python runtime, so the recipient does not install Python or download application dependencies. Choose `AppleSilicon` for M-series Macs or `Intel` for older Intel Macs. See [Native macOS reviewer delivery](MACOS_NATIVE_APP_DELIVERY.md).

The source launcher below remains available for developers and reproducibility work. It is not the recommended non-technical reviewer delivery.

The Windows protected demonstration contains `VitalsSight.exe` and cannot run on macOS. Use the repository's native macOS launcher instead.

## Start the research workflow

1. Download the repository with **Code -> Download ZIP**, then extract the complete folder.
2. Install Python 3.10, 3.11 or 3.12 from [python.org](https://www.python.org/downloads/macos/) if it is not already installed.
3. Double-click `RUN_VITALSSIGHT_MAC.command` in the extracted folder.
4. Keep the Terminal window open. The launcher creates an isolated `.venv-macos` environment, installs the declared dependencies, verifies the pinned face-landmark asset and opens the local VitalsSight page.
5. Press `Control-C` in the Terminal window to stop the local services.

The first start downloads dependencies and normally takes several minutes. Later starts reuse the isolated environment. Application data and logs are stored under:

```text
~/Library/Application Support/VitalsSightResearchDemo
~/Library/Logs/VitalsSight
```

## If macOS blocks the launcher

The launcher is source code rather than a notarized Apple application. If Finder displays a security warning:

1. Control-click `RUN_VITALSSIGHT_MAC.command`.
2. Select **Open**.
3. Confirm **Open** in the macOS dialog.

Do not disable Gatekeeper globally. A signed and notarized `.app` requires an author-controlled Apple Developer identity and is a separate release task.

## What is local

- The UI and API listen only on `127.0.0.1`.
- The deterministic assessment and evidence explanation do not require a remote language model.
- Raw uploaded media are deleted after analysis under the declared workflow.
- Optional Ollama models can run locally but cannot alter the deterministic release/review/retake decision.

This is a research artifact, not a medical device or an autonomous clinical-release system.
