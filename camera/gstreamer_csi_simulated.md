# CSI (Jetson) — simulated GStreamer pipelines

Production Jetson example (device-dependent):

```bash
gst-launch-1.0 nvarguscamerasrc ! 'video/x-raw(memory:NVMM),width=1920,height=1080,framerate=30/1' \
  ! nvvidconv ! 'video/x-raw,format=RGBA' ! fdsink
```

Simulated CSI (file or test pattern → appsink) used by the framework when `EDGE_SIMULATE_CAMERAS=1`:

```bash
gst-launch-1.0 videotestsrc pattern=ball is-live=true ! video/x-raw,width=640,height=480,framerate=30/1 \
  ! videoconvert ! video/x-raw,format=RGB ! fdsink
```

The Python `CameraManager` uses OpenCV or synthetic frames when GStreamer is unavailable (Windows / dev boxes).
