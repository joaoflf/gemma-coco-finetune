import os
from typing import Generator
from pycocotools.coco import COCO
from PIL import Image
from datasets import IterableDataset
from prompts import get_prompt_template


class CocoDataset:
    def __init__(self, split="val2017", data_dir="coco"):
        self.split = split
        self.image_dir = os.path.join(data_dir, split)
        self.annotations_file = os.path.join(
            data_dir, "annotations", f"person_keypoints_{split}.json"
        )
        self.coco = COCO(self.annotations_file)
        self.person_id = self.coco.getCatIds(catNms=["person"])
        self.image_ids = self.coco.getImgIds(catIds=self.person_id)
        self.system_message, self.user_prompt = get_prompt_template()
        print(f"Found {len(self.image_ids)}")

    def get_dataset(self) -> IterableDataset:
        return IterableDataset.from_generator(self._coco_object_generator)

    def _convert_to_oai_format(self, sample: tuple[Image.Image, str]) -> dict:
        image_obj, annotation = sample
        return {
            "messages": [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": self.system_message}],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": self.user_prompt,
                        },
                        {
                            "type": "image",
                            "image": image_obj,
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": annotation}],
                },
            ],
        }

    def _coco_object_generator(self) -> Generator[dict, None, None]:
        for image_id in self.image_ids:
            image_info = self.coco.loadImgs(image_id)[0]
            image_obj = Image.open(
                os.path.join(self.image_dir, image_info["file_name"])
            ).convert("RGB")
            annotation_ids = self.coco.getAnnIds(
                imgIds=image_id, catIds=self.person_id, iscrowd=False
            )
            annotations = self.coco.loadAnns(annotation_ids)
            annotations_with_keypoints = [
                ann
                for ann in annotations
                if "keypoints" in ann and sum(ann["keypoints"][2::3]) > 0
            ]
            yield self._convert_to_oai_format(
                (image_obj, str(annotations_with_keypoints))
            )
