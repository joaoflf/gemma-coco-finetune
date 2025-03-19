mkdir -p coco
wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip -P coco/
wget -c http://images.cocodataset.org/zips/train2017.zip -P coco/
wget -c http://images.cocodataset.org/zips/val2017.zip -P coco

unzip -q coco/annotations_trainval2017.zip -d coco/
unzip -q coco/train2017.zip -d coco/
unzip -q coco/val2017.zip -d coco
