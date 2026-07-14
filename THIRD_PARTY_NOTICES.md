# Third-party notices

## OpenCV face cascade

`src/product/assets/haarcascade_frontalface_default.xml` is distributed by the
OpenCV project and is used only for local video-quality preflight.

- Upstream source: <https://github.com/opencv/opencv/blob/4.x/data/haarcascades/haarcascade_frontalface_default.xml>
- Project: <https://opencv.org/>
- License: BSD 3-Clause, reproduced in the XML file header and available at
  <https://github.com/opencv/opencv/blob/4.x/LICENSE>

Copyright for this asset remains with its upstream authors. No participant data,
images, or video are included with the asset.

## MediaPipe Face Landmarker runtime model

VitalsSight does not redistribute the Face Landmarker model binary. The optional
runtime setup script downloads the model bundle from the official Google AI Edge
host into the ignored `runtime/models` directory and verifies the pinned file
hash before use.

- Official task guide: <https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker>
- Official model URL: <https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task>
- Pinned SHA256: `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`
- MediaPipe project and license: <https://github.com/google-ai-edge/mediapipe>

The upstream model and software remain subject to their provider terms. No
participant media are downloaded by the runtime-asset setup.
