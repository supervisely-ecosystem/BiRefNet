import supervisely as sly
import os
from supervisely.nn.inference import CheckpointInfo
import torch
from transformers import AutoModelForImageSegmentation
from torchvision import transforms
from PIL import Image
import numpy as np


class BiRefNet(sly.nn.inference.SalientObjectSegmentation):
    FRAMEWORK_NAME = "BiRefNet"
    MODELS = "supervisely_integration/src/models.json"
    APP_OPTIONS = "supervisely_integration/src/app_options.yaml"
    INFERENCE_SETTINGS = "supervisely_integration/src/inference_settings.yaml"

    def load_model(
        self,
        model_files: dict,
        model_info: dict,
        model_source: str,
        device: str,
        runtime: str,
    ):
        checkpoint_path = model_files["checkpoint"]
        if sly.is_development():
            checkpoint_path = "." + checkpoint_path
        self.classes = ["object_mask"]
        self.checkpoint_info = CheckpointInfo(
            checkpoint_name=os.path.basename(checkpoint_path),
            model_name=model_info["meta"]["model_name"],
            architecture=self.FRAMEWORK_NAME,
            checkpoint_url=model_info["meta"]["model_files"]["checkpoint"],
            model_source=model_source,
        )
        self.device = torch.device(device)
        self.model = AutoModelForImageSegmentation.from_pretrained(
            checkpoint_path,
            trust_remote_code=True,
        ).eval()
        torch.set_float32_matmul_precision(["high", "highest"][0])
        self.model = self.model.to(self.device)
        self.transform_image = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def binarize_mask(self, mask, threshold=None):
        if threshold is None:
            threshold = 200
        mask[mask < threshold] = 0
        mask[mask >= threshold] = 1
        return mask

    def predict(self, image_path, settings):
        original_image = Image.open(image_path)
        input_image = self.transform_image(original_image).unsqueeze(0).to("cuda")

        with torch.no_grad():
            preds = self.model(input_image)[-1].sigmoid().cpu()

        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        pred_pil = pred_pil.resize(original_image.size)
        mask = np.array(pred_pil)
        threshold = settings.get("pixel_confidence_threshold")
        mask = self.binarize_mask(mask, threshold)
        return [sly.nn.PredictionMask(class_name=self.classes[0], mask=mask)]
