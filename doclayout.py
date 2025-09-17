import abc
import os.path

import cv2
import numpy as np
import ast

from pdf2zh.config import ConfigManager


class DocLayoutModel(abc.ABC):
    @staticmethod
    def load_onnx():
        raise RuntimeError("ONNX backend has been removed. Use Torch backend instead.")

    @staticmethod
    def load_available():
        return TorchModel.from_pretrained()

    @property
    @abc.abstractmethod
    def stride(self) -> int:
        """Stride of the model input."""
        pass

    @abc.abstractmethod
    def predict(self, image, imgsz=1024, **kwargs) -> list:
        """
        Predict the layout of a document page.

        Args:
            image: The image of the document page.
            imgsz: Resize the image to this size. Must be a multiple of the stride.
            **kwargs: Additional arguments.
        """
        pass


class YoloResult:
    """Helper class to store detection results from ONNX model."""

    def __init__(self, boxes, names):
        self.boxes = [YoloBox(data=d) for d in boxes]
        self.boxes.sort(key=lambda x: x.conf, reverse=True)
        self.names = names


class YoloBox:
    """Helper class to store detection results from ONNX model."""

    def __init__(self, data):
        self.xyxy = data[:4]
        self.conf = data[-2]
        self.cls = data[-1]


class TorchModel(DocLayoutModel):
    def __init__(self, weights_path: str | None = None, model_name: str | None = None):
        # Lazy import to avoid hard dependency if unused
        from ultralytics import YOLO

        # Use a basic YOLOv8 model if no weights specified
        if not weights_path:
            # Use YOLOv8n (nano) as a lightweight default for document layout detection
            self._model_name = "yolov8n.pt"
            self._weights_path = None
            self.model = YOLO(self._model_name)
        else:
            if not os.path.exists(weights_path):
                raise FileNotFoundError(
                    f"DocLayout weights not found at '{weights_path}'. Please provide a valid .pt file."
                )
            self._model_name = model_name or os.path.basename(weights_path)
            self._weights_path = weights_path
            self.model = YOLO(weights_path)

        # names: dict[int, str]
        names = self.model.names
        if isinstance(names, dict):
            self._names = [names[i] for i in sorted(names.keys())]
        else:
            self._names = list(names)

        # stride may not be exposed; default to 32 which is typical for YOLO
        try:
            # model.model.stride is a tensor in Ultralytics
            s = getattr(getattr(self.model, "model", None), "stride", None)
            self._stride = int(np.max(s.cpu().numpy())) if s is not None else 32
        except Exception:
            self._stride = 32

        # Build a mapping from class id to name for convenience
        self._id_to_name = {idx: name for idx, name in enumerate(self._names)}

    @staticmethod
    def from_pretrained(weights_path: str | None = None, model_name: str | None = None):
        return TorchModel(weights_path=weights_path, model_name=model_name)

    @property
    def stride(self):
        return self._stride

    def resize_and_pad_image(self, image, new_shape):
        """
        Resize and pad the image to the specified size, ensuring dimensions are multiples of stride.

        Parameters:
        - image: Input image
        - new_shape: Target size (integer or (height, width) tuple)
        - stride: Padding alignment stride, default 32

        Returns:
        - Processed image
        """
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        h, w = image.shape[:2]
        new_h, new_w = new_shape

        # Calculate scaling ratio
        r = min(new_h / h, new_w / w)
        resized_h, resized_w = int(round(h * r)), int(round(w * r))

        # Resize image
        image = cv2.resize(
            image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR
        )

        # Calculate padding size and align to stride multiple
        pad_w = (new_w - resized_w) % self.stride
        pad_h = (new_h - resized_h) % self.stride
        top, bottom = pad_h // 2, pad_h - pad_h // 2
        left, right = pad_w // 2, pad_w - pad_w // 2

        # Add padding
        image = cv2.copyMakeBorder(
            image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )

        return image

    def scale_boxes(self, img1_shape, boxes, img0_shape):
        """
        Rescales bounding boxes (in the format of xyxy by default) from the shape of the image they were originally
        specified in (img1_shape) to the shape of a different image (img0_shape).

        Args:
            img1_shape (tuple): The shape of the image that the bounding boxes are for,
                in the format of (height, width).
            boxes (torch.Tensor): the bounding boxes of the objects in the image, in the format of (x1, y1, x2, y2)
            img0_shape (tuple): the shape of the target image, in the format of (height, width).

        Returns:
            boxes (torch.Tensor): The scaled bounding boxes, in the format of (x1, y1, x2, y2)
        """

        # Calculate scaling ratio
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])

        # Calculate padding size
        pad_x = round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1)
        pad_y = round((img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1)

        # Remove padding and scale boxes
        boxes[..., :4] = (boxes[..., :4] - [pad_x, pad_y, pad_x, pad_y]) / gain
        return boxes

    def predict(self, image, imgsz=1024, **kwargs):
        # Run inference via Ultralytics which handles resize/letterbox internally
        results = self.model.predict(
            source=image, imgsz=imgsz, conf=kwargs.get("conf", 0.25), verbose=False
        )
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return [YoloResult(boxes=np.zeros((0, 6), dtype=np.float32), names=self._names)]

        # Extract boxes as numpy [x1, y1, x2, y2, conf, cls]
        xyxy = r.boxes.xyxy.cpu().numpy()
        conf = r.boxes.conf.cpu().numpy().reshape(-1, 1)
        cls = r.boxes.cls.cpu().numpy().reshape(-1, 1)
        preds = np.concatenate([xyxy, conf, cls], axis=1)
        return [YoloResult(boxes=preds, names=self._names)]


class ModelInstance:
    value: DocLayoutModel = None
