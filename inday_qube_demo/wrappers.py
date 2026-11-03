"""Gymnasium wrappers for recording and showing rendered episodes."""

import os

import gymnasium as gym
from gymnasium.wrappers import RecordVideo


class RenderCache(gym.Wrapper):
    """Render once per step/reset; further `render()` calls return the cached frame."""

    def __init__(self, env):
        super().__init__(env)
        self._frame = None

    def reset(self, **kwargs):
        self._frame = None
        return super().reset(**kwargs)

    def step(self, action):
        self._frame = None
        return super().step(action)

    def render(self):
        if self._frame is None:
            self._frame = super().render()
        return self._frame


class StreamingRecordVideo(RecordVideo):
    """RecordVideo that streams frames to ffmpeg instead of keeping them in memory."""

    _writer = None

    def _capture_frame(self):
        from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter

        frame = self.env.render()
        if self._writer is None:
            path = os.path.join(self.video_folder, f"{self._video_name}.mp4")
            size = (frame.shape[1], frame.shape[0])
            self._writer = FFMPEG_VideoWriter(path, size, self.frames_per_sec)
        self._writer.write_frame(frame)

    def stop_recording(self):
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        self.recording = False
        self._video_name = None
