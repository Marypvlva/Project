from pathlib import Path

import cv2
import numpy as np
import torch


class VideoProcessor:
    """
    Обработка MP4-видео лазерного процесса.

    Логика:
    - видео уже заранее сконвертированы из CINE в MP4;
    - все кадры сохранены;
    - MP4 могут иметь искусственный FPS, например 50;
    - поэтому для выбора кадра используется не абсолютное время,
      а относительная позиция внутри видео от 0 до 1.

    Например:
        position = 0.10 -> 10% видео
        position = 0.50 -> середина видео
        position = 0.90 -> 90% видео

    Итоговый кадр:
        torch.Tensor
        shape = [3, 128, 128]
        dtype = torch.float32
        значения = [0, 1]
    """

    def __init__(
        self,
        frame_size=(128, 128),
        position_step=0.02
    ):
        """
        Parameters
        ----------
        frame_size : tuple
            Размер кадра для нейросети.
            По умолчанию (128, 128).

        position_step : float
            Шаг по нормированной позиции видео.

            Например:
                0.02 -> 2%, 4%, 6%, ...
                0.05 -> 5%, 10%, 15%, ...
                0.10 -> 10%, 20%, 30%, ...
        """

        self.frame_size = frame_size
        self.position_step = position_step

        # Кэш метаданных видео.
        # Чтобы не открывать одно и то же MP4
        # заново только ради FPS/числа кадров.
        self._video_info_cache = {}

    def get_video_info(self, video_path):
        """
        Получить информацию об MP4.

        Возвращает:
            {
                "fps": ...,
                "total_frames": ...,
                "width": ...,
                "height": ...,
                "duration": ...
            }
        """

        video_path = Path(video_path)

        if not video_path.exists():
            raise FileNotFoundError(
                f"Видео не найдено: {video_path}"
            )

        cache_key = str(
            video_path.resolve()
        )

        if cache_key in self._video_info_cache:
            return self._video_info_cache[
                cache_key
            ]

        cap = cv2.VideoCapture(
            str(video_path)
        )

        if not cap.isOpened():
            raise RuntimeError(
                f"Не удалось открыть MP4: {video_path}"
            )

        fps = float(
            cap.get(
                cv2.CAP_PROP_FPS
            )
        )

        total_frames = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        width = int(
            cap.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        height = int(
            cap.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        cap.release()

        if fps <= 0:
            raise ValueError(
                f"Некорректный FPS: {fps}"
            )

        if total_frames <= 0:
            raise ValueError(
                "Количество кадров должно быть больше нуля."
            )

        duration = (
            total_frames / fps
        )

        info = {
            "fps": fps,
            "total_frames": total_frames,
            "width": width,
            "height": height,
            "duration": duration
        }

        self._video_info_cache[
            cache_key
        ] = info

        return info

    def get_position_grid(self):
        """
        Построить сетку относительных позиций по видео.

        Например при position_step = 0.1:

            0.1
            0.2
            0.3
            ...
            0.9

        Значения лежат в диапазоне (0, 1).
        """

        if self.position_step <= 0:
            raise ValueError(
                "position_step должен быть больше нуля."
            )

        if self.position_step >= 1:
            raise ValueError(
                "position_step должен быть меньше 1."
            )

        positions = np.arange(
            self.position_step,
            1.0,
            self.position_step,
            dtype=np.float32
        )

        return positions

    @staticmethod
    def position_to_frame_index(
        position,
        total_frames
    ):
        """
        Перевести относительную позицию
        внутри видео в номер кадра.

        Например:

            position = 0.5
            total_frames = 2000

        Получаем примерно:
            frame_idx = 1000
        """

        if not 0.0 <= position <= 1.0:
            raise ValueError(
                "position должна находиться "
                "в диапазоне от 0 до 1."
            )

        if total_frames <= 0:
            raise ValueError(
                "total_frames должен быть больше нуля."
            )

        frame_idx = int(
            round(
                position
                * (total_frames - 1)
            )
        )

        return frame_idx

    def read_mp4_frame(
        self,
        video_path,
        position
    ):
        """
        Прочитать один кадр MP4
        по относительной позиции в видео.

        Parameters
        ----------
        video_path:
            путь к MP4

        position:
            относительная позиция от 0 до 1

            0.1 -> 10% видео
            0.5 -> середина
            0.9 -> 90%

        Returns
        -------
        numpy.ndarray
            исходный кадр OpenCV
            shape = [H, W, 3]
            формат = BGR
            dtype = uint8
        """

        video_path = Path(
            video_path
        )

        info = self.get_video_info(
            video_path
        )

        frame_idx = (
            self.position_to_frame_index(
                position=position,
                total_frames=info[
                    "total_frames"
                ]
            )
        )

        cap = cv2.VideoCapture(
            str(video_path)
        )

        if not cap.isOpened():
            raise RuntimeError(
                f"Не удалось открыть MP4: {video_path}"
            )

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_idx
        )

        ret, frame = cap.read()

        cap.release()

        if not ret or frame is None:
            raise RuntimeError(
                f"Не удалось прочитать кадр "
                f"{frame_idx} "
                f"для позиции {position:.3f}"
            )

        return frame

    def preprocess_frame(
        self,
        frame
    ):
        """
        Подготовить MP4-кадр для нейросети.

        Pipeline:

            BGR uint8
            [H, W, 3]
                ↓
            RGB
                ↓
            resize
                ↓
            [128, 128, 3]
                ↓
            float32 / 255
                ↓
            [0, 1]
                ↓
            HWC -> CHW
                ↓
            Tensor [3, 128, 128]
        """

        if frame is None:
            raise ValueError(
                "Получен пустой кадр."
            )

        if (
            frame.ndim != 3
            or frame.shape[2] != 3
        ):
            raise ValueError(
                f"Ожидался цветной кадр "
                f"[H, W, 3], "
                f"получено: {frame.shape}"
            )

        # OpenCV читает MP4 как BGR.
        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        frame = cv2.resize(
            frame,
            self.frame_size,
            interpolation=cv2.INTER_AREA
        )

        frame = (
            frame.astype(
                np.float32
            )
            / 255.0
        )

        frame = np.clip(
            frame,
            0.0,
            1.0
        )

        # HWC -> CHW
        frame = torch.from_numpy(
            frame
        ).permute(
            2,
            0,
            1
        )

        frame = frame.contiguous()

        return frame.float()

    def get_frame(
        self,
        video_path,
        position
    ):
        """
        Получить готовый кадр для модели.

        Parameters
        ----------
        video_path:
            путь к MP4

        position:
            относительная позиция видео
            в диапазоне 0..1

        Returns
        -------
        torch.Tensor
            shape = [3, 128, 128]
            dtype = torch.float32
            значения = [0, 1]
        """

        frame = self.read_mp4_frame(
            video_path,
            position
        )

        frame = self.preprocess_frame(
            frame
        )

        return frame
