**MSD: Mobile Phone Defect Segmentation Dataset** is a dataset for instance segmentation, semantic segmentation, and object detection tasks. It is used in the surface defect detection domain. 

The dataset consists of 1220 images with 3104 labeled objects belonging to 3 different classes including *stain*, *oil*, and *scratch*.

Images in the MSD dataset have pixel-level instance segmentation annotations. Due to the nature of the instance segmentation task, it can be automatically transformed into a semantic segmentation (only one mask for every class) or object detection (bounding boxes for every object) tasks. There are 20 (2% of the total) unlabeled images (i.e. without annotations). There are no pre-defined <i>train/val/test</i> splits in the dataset. Additionally, images without defects are marked with ***good*** tag. The dataset was released in 2021 by the <span style="font-weight: 600; color: grey; border-bottom: 1px dashed #d3d3d3;">Peking University</span>.

<img src="https://github.com/dataset-ninja/mobile-phone-defect-segmentation/raw/main/visualizations/poster.png">
