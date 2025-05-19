import dtlpy as dl
import logging
import torch
import PIL

from torchvision.ops import box_convert
from groundingdino.util.inference import load_model, load_image, predict, annotate

logger = logging.getLogger('GroundingDINOAdapter')

# set max image size
PIL.Image.MAX_IMAGE_PIXELS = 933120000


class Adapter(dl.BaseModelAdapter):
    """
    Model Adapter class for loading and using the YOLOWorld model.
    """

    def load(self, local_path, **kwargs):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        model = load_model(
            "/tmp/GroundingDINO/config/GroundingDINO_SwinT_OGC.py",
            "/tmp/GroundingDINO/weights/groundingdino_swint_ogc.pth",
            device=self.device,
        )
        logger.info(f"Model loaded successfully, Device: {self.device}")
        self.model = model

    def prepare_item_func(self, item):
        image_source, image = load_image(item.download(overwrite=True))
        return image_source, image, item

    def predict(self, batch, **kwargs):
        box_threshold = self.model_entity.configuration.get('box_threshold', 0.35)
        text_threshold = self.model_entity.configuration.get('text_threshold', 0.25)

        batch_annotations = list()
        for image_source, image, item in batch:
            model_labels = list(self.model_entity.label_to_id_map.keys())
            if len(model_labels) > 0:
                text_prompt = ' . '.join(model_labels)
            else:
                dataset_labels = list(item.dataset.labels_flat_dict.keys())
                text_prompt = ' . '.join(dataset_labels)
            boxes, logits, phrases = predict(
                model=self.model,
                device=self.device,
                image=image,
                caption=text_prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
            )

            # annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
            # cv2.imwrite("annotated_image.jpg", annotated_frame)
            h, w, _ = image_source.shape
            boxes = boxes * torch.Tensor([w, h, w, h])
            xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
            image_annotations = dl.AnnotationCollection()
            for bb, lbl, conf in zip(xyxy, phrases, logits):
                xyxy = bb.squeeze()
                image_annotations.add(
                    annotation_definition=dl.Box(
                        left=float(xyxy[0]), top=float(xyxy[1]), right=float(xyxy[2]), bottom=float(xyxy[3]), label=lbl
                    ),
                    model_info={
                        'name': self.model_entity.name,
                        'model_id': self.model_entity.id,
                        'confidence': float(f"{conf:.2f}"),
                    },
                )
            batch_annotations.append(image_annotations)
        return batch_annotations
