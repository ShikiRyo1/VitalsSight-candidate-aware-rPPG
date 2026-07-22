from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time


VERSION = "0.2.0-macos.1"
BUNDLE_ID = "org.vitalsight.research-demo"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (FileNotFoundError, subprocess.SubprocessError):
        return "unknown"


def build_metadata(root: Path) -> dict[str, str]:
    return {
        "version": VERSION,
        "commit": git_value(root, "rev-parse", "HEAD"),
        "tree": git_value(root, "rev-parse", "HEAD^{tree}"),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "platform": "macOS",
        "architecture": platform.machine(),
        "package_scope": "Self-contained VitalsSight macOS research demonstration",
    }


def make_icon(root: Path, output: Path) -> Path:
    from PIL import Image, ImageDraw

    iconset = output / "VitalsSight.iconset"
    if iconset.exists():
        shutil.rmtree(iconset)
    iconset.mkdir(parents=True)
    master = Image.new("RGBA", (1024, 1024), (0, 0, 0, 0))
    draw = ImageDraw.Draw(master)
    draw.rounded_rectangle((28, 28, 996, 996), radius=224, fill="#EEF6F8", outline="#39778A", width=30)
    draw.rounded_rectangle((158, 174, 866, 850), radius=150, fill="#FFFFFF", outline="#6A9F98", width=20)
    points = [(214, 532), (340, 532), (405, 388), (493, 666), (578, 463), (646, 532), (810, 532)]
    draw.line(points, fill="#39778A", width=46, joint="curve")
    draw.ellipse((455, 494, 541, 580), fill="#C97167")
    sizes = {
        "icon_16x16.png": 16,
        "icon_16x16@2x.png": 32,
        "icon_32x32.png": 32,
        "icon_32x32@2x.png": 64,
        "icon_128x128.png": 128,
        "icon_128x128@2x.png": 256,
        "icon_256x256.png": 256,
        "icon_256x256@2x.png": 512,
        "icon_512x512.png": 512,
        "icon_512x512@2x.png": 1024,
    }
    for name, size in sizes.items():
        master.resize((size, size), Image.Resampling.LANCZOS).save(iconset / name)
    target = output / "VitalsSight.icns"
    subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(target)], check=True)
    return target


def add_data_args(root: Path) -> list[str]:
    entries: list[tuple[Path, str]] = []
    for name in ("configs", "knowledge", ".streamlit"):
        path = root / name
        if path.exists():
            entries.append((path, name))
    reproducibility = root / "reproducibility"
    if reproducibility.exists():
        entries.append((reproducibility, "reproducibility"))
    model = root / "runtime" / "models" / "face_landmarker.task"
    if not model.is_file():
        raise FileNotFoundError(f"Pinned MediaPipe model is missing: {model}")
    entries.append((model, "runtime/models"))
    for name, destination in (
        ("THIRD_PARTY_NOTICES.md", "."),
        ("packaging/build_metadata.json", "packaging"),
    ):
        path = root / name
        if path.is_file():
            entries.append((path, destination))
    args: list[str] = []
    for source, destination in entries:
        args.extend(["--add-data", f"{source}{os.pathsep}{destination}"])
    return args


def write_delivery_notes(folder: Path, metadata: dict[str, str]) -> None:
    chinese = """VitalsSight macOS 研究演示版\n\n1. 先确认电脑类型：Apple 菜单 > 关于本机。\n2. Apple M1/M2/M3/M4/M5 请选择 AppleSilicon 包；旧款 Intel Mac 请选择 Intel 包。\n3. 解压完整 ZIP，不要直接在压缩包预览中运行。\n4. 第一次启动：按住 Control 点击 VitalsSight.app，选择“打开”，再确认“打开”。以后可直接双击。\n5. 等待启动窗口显示 VitalsSight is ready；浏览器会自动打开。\n6. 软件在本机 127.0.0.1 运行，不需要安装 Python，也不依赖作者电脑。\n\n如果启动失败，日志位于：~/Library/Logs/VitalsSight\n这是研究演示软件，不是医疗器械，也不提供自主临床诊断或放行。\n"""
    english = """VitalsSight macOS Research Demo\n\n1. Check Apple menu > About This Mac. Use AppleSilicon for M-series Macs and Intel for older Intel Macs.\n2. Extract the complete ZIP before opening the application.\n3. On first launch, Control-click VitalsSight.app, choose Open, then confirm Open. Later launches can use a normal double-click.\n4. Wait for the launcher to report VitalsSight is ready; the browser opens automatically.\n5. The application runs locally on 127.0.0.1, requires no Python installation, and does not depend on the author's computer.\n\nLogs: ~/Library/Logs/VitalsSight\nThis is research software, not a medical device or autonomous clinical-release system.\n"""
    (folder / "README_FIRST_CN.txt").write_text(chinese, encoding="utf-8")
    (folder / "README_FIRST_EN.txt").write_text(english, encoding="utf-8")
    (folder / "PACKAGE_INFO.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def repair_macos_cv2_crosslink(app: Path) -> dict[str, object]:
    """Ensure the OpenCV loader in Resources can see its Frameworks extension."""
    contents = app / "Contents"
    loaders = sorted((contents / "Resources").glob("cv2/__init__.py"))
    extensions = sorted((contents / "Frameworks").glob("cv2/cv2*.so"))
    if len(loaders) != 1 or len(extensions) != 1:
        raise RuntimeError(
            "Unexpected OpenCV bundle layout: "
            f"loaders={[str(path) for path in loaders]}, "
            f"extensions={[str(path) for path in extensions]}"
        )
    loader_dir = loaders[0].parent
    extension = extensions[0]
    target = loader_dir / extension.name
    expected = os.path.relpath(extension, loader_dir)
    if target.is_symlink() and os.readlink(target) != expected:
        target.unlink()
    elif target.exists() and not target.is_symlink():
        if target.samefile(extension):
            return {"loader": str(loaders[0]), "extension": str(extension), "crosslink": str(target)}
        target.unlink()
    if not target.exists():
        target.symlink_to(expected)
    if not target.exists() or not target.samefile(extension):
        raise RuntimeError(f"OpenCV extension crosslink is invalid: {target} -> {expected}")
    return {
        "loader": str(loaders[0]),
        "extension": str(extension),
        "crosslink": str(target),
        "crosslink_target": os.readlink(target),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    if sys.platform != "darwin":
        raise SystemExit("The macOS application must be built on macOS")
    root = args.root.resolve()
    output = (args.output_root or root / "build" / "macos").resolve()
    work = output / "work"
    dist = output / "dist"
    delivery = output / "delivery"
    for path in (work, dist, delivery):
        path.mkdir(parents=True, exist_ok=True)

    metadata = build_metadata(root)
    metadata_path = root / "packaging" / "build_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    icon = make_icon(root, output)
    pyinstaller_args = [
        str(root / "packaging" / "macos_launcher.py"),
        "--noconfirm",
        "--clean",
        "--onedir",
        "--windowed",
        "--name",
        "VitalsSight",
        "--icon",
        str(icon),
        "--osx-bundle-identifier",
        BUNDLE_ID,
        "--distpath",
        str(dist),
        "--workpath",
        str(work),
        "--specpath",
        str(output),
        "--paths",
        str(root),
        "--hidden-import",
        "app.api_server",
        "--hidden-import",
        "app.product_console",
        "--hidden-import",
        "cv2",
        "--hidden-import",
        "multipart",
        "--collect-all",
        "streamlit",
        "--collect-all",
        "mediapipe",
        "--collect-all",
        "plotly",
        "--collect-submodules",
        "app",
        "--collect-submodules",
        "src",
        "--collect-submodules",
        "uvicorn",
        "--exclude-module",
        "torch",
        "--exclude-module",
        "tensorflow",
        "--exclude-module",
        "faster_whisper",
        "--exclude-module",
        "ctranslate2",
    ]
    pyinstaller_args.extend(add_data_args(root))
    from PyInstaller.__main__ import run

    run(pyinstaller_args)
    app = dist / "VitalsSight.app"
    if not app.is_dir():
        raise FileNotFoundError(f"PyInstaller did not create {app}")
    cv2_layout = repair_macos_cv2_crosslink(app)
    (output / "opencv_bundle_layout.json").write_text(
        json.dumps(cv2_layout, indent=2), encoding="utf-8"
    )
    subprocess.run(["codesign", "--force", "--deep", "--sign", "-", str(app)], check=True)
    subprocess.run(["codesign", "--verify", "--deep", "--strict", "--verbose=2", str(app)], check=True)

    architecture = platform.machine()
    label = "AppleSilicon" if architecture == "arm64" else "Intel"
    folder = delivery / f"VitalsSight_macOS_{label}_{VERSION}"
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    shutil.copytree(app, folder / app.name, symlinks=True)
    write_delivery_notes(folder, metadata)
    target = delivery / f"VitalsSight_macOS_{label}_{VERSION}.zip"
    if target.exists():
        target.unlink()
    subprocess.run(
        ["ditto", "-c", "-k", "--sequesterRsrc", "--keepParent", str(folder), str(target)],
        check=True,
    )
    checksum = sha256_file(target)
    (delivery / f"SHA256_{label}.txt").write_text(f"{checksum}  {target.name}\n", encoding="ascii")
    print(
        json.dumps(
            {
                "app": str(app),
                "zip": str(target),
                "sha256": checksum,
                "metadata": metadata,
                "opencv_bundle_layout": cv2_layout,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
