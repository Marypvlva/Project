from pathlib import Path

import cv2
import numpy as np
import torch


class VideoProcessor:
    """
    Чтение и предобработка MP4-видео лазерного процесса.

    Важная идея:
    кадр выбирается не по времени, а по относительной позиции внутри видео.
    Это позволяет корректно работать с MP4, у которых FPS мог измениться
    при конвертации из исходного формата.

    На выходе get_frame():
        torch.Tensor [3, H, W]
        dtype=torch.float32
        значения в диапазоне [0, 1]
    """

    def __init__(self, frame_size=(128, 128), position_step=0.02):
        if len(frame_size) != 2:
            raise ValueError("frame_size должен иметь вид (width, height).")

        width, height = frame_size
        if width <= 0 or height <= 0:
            raise ValueError("Размер кадра должен быть положительным.")

        if not 0.0 < position_step < 1.0:
            raise ValueError("position_step должен находиться в диапазоне (0, 1).")

        self.frame_size = (int(width), int(height))
        self.position_step = float(position_step)

        # Здесь хранятся только небольшие метаданные видео, а не сами кадры.
        self._video_info_cache = {}

    def get_video_info(self, video_path):
        """Получить FPS, число кадров, размер и длительность MP4."""
        video_path = Path(video_path)

        if not video_path.exists():
            raise FileNotFoundError(f"Видео не найдено: {video_path}")
        if not video_path.is_file():
            raise ValueError(f"Ожидался файл видео: {video_path}")

        cache_key = str(video_path.resolve())
        if cache_key in self._video_info_cache:
            return self._video_info_cache[cache_key]

        cap = cv2.VideoCapture(str(video_path))
        try:
            if not cap.isOpened():
                raise RuntimeError(f"Не удалось открыть MP4: {video_path}")

            fps = float(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        finally:
            cap.release()

        if not np.isfinite(fps) or fps <= 0:
            raise ValueError(f"Некорректный FPS: {fps}")
        if total_frames <= 0:
            raise ValueError("Количество кадров должно быть больше нуля.")
        if width <= 0 or height <= 0:
            raise ValueError(f"Некорректный размер видео: {width}x{height}")

        info = {
            "fps": fps,
            "total_frames": total_frames,
            "width": width,
            "height": height,
            "duration": total_frames / fps,
        }
        self._video_info_cache[cache_key] = info
        return info

    def get_position_grid(self):
        """
        Построить одинаковую для всех видео сетку относительных позиций.

        position_step=0.02 -> 0.02, 0.04, ..., 0.98.
        Нулевой и последний кадры намеренно не используются.
        """
        positions = np.arange(
            self.position_step,
            1.0,
            self.position_step,
            dtype=np.float32,
        )

        # Защита от численных эффектов np.arange около 1.0.
        positions = positions[(positions > 0.0) & (positions < 1.0)]

        if len(positions) == 0:
            raise ValueError("Не удалось построить сетку позиций.")

        return positions

    @staticmethod
    def position_to_frame_index(position, total_frames):
        """Преобразовать позицию 0..1 в индекс кадра 0..total_frames-1."""
        position = float(position)

        if not np.isfinite(position) or not 0.0 <= position <= 1.0:
            raise ValueError("position должна находиться в диапазоне [0, 1].")
        if total_frames <= 0:
            raise ValueError("total_frames должен быть больше нуля.")

        frame_idx = int(round(position * (total_frames - 1)))
        return max(0, min(frame_idx, total_frames - 1))

    def read_mp4_frame(self, video_path, position):
        """Прочитать один исходный BGR-кадр по относительной позиции."""
        video_path = Path(video_path)
        info = self.get_video_info(video_path)

        frame_idx = self.position_to_frame_index(
            position=position,
            total_frames=info["total_frames"],
        )

        cap = cv2.VideoCapture(str(video_path))
        try:
            if not cap.isOpened():
                raise RuntimeError(f"Не удалось открыть MP4: {video_path}")

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
        finally:
            cap.release()

        if not ret or frame is None:
            raise RuntimeError(
                f"Не удалось прочитать кадр {frame_idx} "
                f"для позиции {float(position):.3f}: {video_path}"
            )

        return frame

    def preprocess_frame(self, frame):
        """
        BGR uint8 [H,W,3] -> RGB -> resize -> float32 [0,1] -> CHW Tensor.
        """
        if frame is None:
            raise ValueError("Получен пустой кадр.")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                f"Ожидался цветной кадр [H, W, 3], получено: {frame.shape}"
            )

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(
            frame,
            self.frame_size,
            interpolation=cv2.INTER_AREA,
        )
        frame = frame.astype(np.float32) / 255.0
        frame = np.clip(frame, 0.0, 1.0)

        frame = torch.from_numpy(frame).permute(2, 0, 1).contiguous()
        return frame.float()

    def get_frame(self, video_path, position):
        """Получить полностью подготовленный кадр для нейросети."""
        frame = self.read_mp4_frame(video_path, position)
        return self.preprocess_frame(frame)
