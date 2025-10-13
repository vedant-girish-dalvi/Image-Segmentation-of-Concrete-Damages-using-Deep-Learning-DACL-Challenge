# dacl10k Dataset

**[Twitter/X](https://twitter.com/dacl_ai) | [WACV](https://wacv2024.thecvf.com/workshops/) | [Toolkit](https://github.com/phiyodr/dacl10k-toolkit) | [arXiv](https://arxiv.org/abs/2309.00460) | [dacl.ai](https://dacl.ai/)**

dacl10k stands for *damage classification 10k images* and is a **multi-label semantic segmentation** dataset for 19 classes (13 damages and 6 objects) present on bridges. 

This dataset is used in the challenge associated with the **"1st Workshop on Vision-Based Structural Inspections in Civil Engineering" at [WACV2024](https://wacv2024.thecvf.com/workshops/).**

# Version

v2 (20230811)

# License

[(CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/)


# Citation

* Link to the paper: [arXiv](https://arxiv.org/abs/2309.00460)
* Please cite as:


```
@misc{flotzinger2023dacl10k,
      author={Johannes Flotzinger and Philipp J. Rösch and Thomas Braml},
      title={dacl10k: Benchmark for Semantic Bridge Damage Segmentation}, 
      year={2023},
      eprint={https://arxiv.org/abs/2309.00460},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

# Description

### Task

Multi-label semantic segmentation

* Multi-label: Each pixel can be associated with several classes, e.g. a surface can have Rust and Crack (2 damages) or Rust on Drainage (1 damage and 1 object).
* Semantic segmentation: Pixel detailed annotation of damages and objects.
* We are not interested in "instances". If several polygons of the same class overlap, they are merged into one mask. 

For evaluation we use [mean Intersection over Union (mIoU)](https://github.com/qubvel/segmentation_models.pytorch/blob/67aceba4b832a36cb99c2184a06a204ba43c4ea1/segmentation_models_pytorch/utils/metrics.py#L6).



### Folder structure


The folder has following format:

```bash
├── README.md
├── LICENSE.md
├── annotations
│   ├── train (n=6.935)
│   └── validation (n=975)
└── images
    ├── train (n=6.935)
    ├── validation (n=975)
    └── testdev (n=1012)
```


### Annotation files

Annotation files originate from [labelme](https://github.com/wkentaro/labelme/tree/main/examples/semantic_segmentation)-format and are slightly adjusted.
One JSON file looks like this:

```javascript
{'imageWidth': 1280,
 'imageHeight': 960,
 'imageName': 'dacl10k_v2_validation_0012.jpg',
 'imagePath': 'images/train/dacl10k_v2_validation_0012.jpg', 
 'split': 'validation',
 'dacl10k_version': 'v2',
 'shapes': [{'label': 'Rust',
   'points': [...],
   'shape_type': 'polygon'},
  {'label': 'Rust',
   'points': [...],
   'shape_type': 'polygon'},
  {'label': 'Drainage',
   'points': [[581.5714285714286, 410.2857142857142],
    [555.8571428571428, 407.4285714285714],
    [524.4285714285713, 435.99999999999994],
    [507.2857142857142, 486.0],
    [502.99999999999994, 531.7142857142857],
    [550.1428571428571, 574.5714285714286],
    [578.7142857142857, 593.1428571428571],
    [612.9999999999999, 560.2857142857142],
    [625.8571428571428, 508.8571428571428],
    [622.9999999999999, 447.42857142857133],
    [605.8571428571428, 420.2857142857142]],
   'shape_type': 'polygon'}]}
```

Explanation:

* `imageWidth`, `imageHeight`: Image width and height (many different image sizes exists).
* `imageName`, `imagePath`, `split`: Corresponding image name and full path, as well the corresponding dataset split. 
* `dacl10k_version`: The first version of the arXiv paper uses v1 and the challenge uses v2.
* `shapes` (list of dictionaries): Each dictionary contains
  * `label` (str): Name of the class
  * `points` (list of lists): List of edge points of the polygone (x, y). Origin in top-left corner.
  * `shape_type`: Always 'polygone' (originates from labelme)


### Labels/Classes 

Each polygone can have one of 19 classes. For detailed explanation of each class please see **[Appendix A.3. "Class description"](https://arxiv.org/abs/2309.00460)**.

* 13 damage classes: Crack, Alligator Crack (ACrack), Wetspot, Efflorescence, Rust, Rockpocket, Hollowareas, Cavity, Spalling, Graffiti, Weathering, Restformwork, Exposed Rebars (ExposedRebars), 
* 6 object classes: Bearing, Expansion Joint (EJoint), Drainage, Protective Equipment (PEquipment), Joint Tape (JTape), Washouts/Concrete corrosion (WConccor)

### Toolkit

The [dacl10k-toolkit](https://github.com/phiyodr/dacl10k-toolkit) should act as an entry point for simple data usage in Python and PyTorch for the dacl10k dataset.# dacl_challenge
