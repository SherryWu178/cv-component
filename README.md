# cv-component

This repo contains the code to train  an resnet18-based image-classfication model for electronic component.

## Folder structure
```
.
├── cv-component (this repo)
│   ├── classfication
|       ├── ...
│   └── checkpoints
|       ├── ...
├── data
│   ├── dataset-1-folder
│   ├── dataset-2-folder
│   └── ...
```
In **classification** folder,

 - train.py contains the high-level code for training the model & defines model architecture
 - trainer.py contains the details for training
 - utils.py contains the code for various helper function for dataloading, argument parsing & validation set evaluation
 - voc_dataset.py  contains the code for data ladoing and augmentation. Note you will need to change the definition of `classes = ...` in `VOCDataset` class to the actual folder name used in the dataset folder.
 - inference_camera contains the logic for single-image prediction

Other files are not in use now but kept for future reference.

## Colab Notebook
The following files demonstrate how to use google colab for data preparation, model training & inference.
**videos_to_frames.ipynb** contains the code to extract frames from videos.
**train.ipynb** contains the examples of how to train models.
**Real-time Inference.ipynb** contains the examples of how to train models.


