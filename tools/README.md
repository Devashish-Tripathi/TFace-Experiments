## Introduction

Face Recognition module consists of several parts: 1.datasets and samplers, 2. backbone model zoo, 3. our proposed methods for face recognition, 4. test protocols of evaluation results and model latency.


## Getting Started

### Train Data

The training dataset is organized in tfrecord format for efficiency. The raw data of all face images are saved in tfrecord files, and each dataset has a corresponding index file(each line includes tfrecord_name, trecord_index offset, label). 

The `Dataset` class will parse the index file to gather image data and label for training. This form of dataset is convenient for reorganization in data cleaning(do not reproduce tfrecord, just reproduce the index file).

1. Convert raw image to tfrecords, generate a new data dir including some tfrecord files and a index_map file
``` bash
python3 tools/img2tfrecord.py --help
usage: img2tfrecord.py [-h] --img_list IMG_LIST --pts_list PTS_LIST
                       --tfrecords_name TFRECORDS_NAME

imgs to tfrecord

optional arguments:
  -h, --help            show this help message and exit
  --img_list IMG_LIST   path to the image file (default: None)
  --pts_list PTS_LIST   path to 5p list (default: None)
  --tfrecords_name TFRECORDS_NAME
                        path to the output of tfrecords dir path (default:
                        TFR-MS1M)
```

2. Convert old index file(each line includes image path, label) to new index file
``` bash
python3 tools/convert_new_index.py --help
usage: convert_new_index.py [-h] --old OLD --tfr_index TFR_INDEX --new NEW

convert training index file

optional arguments:
  -h, --help            show this help message and exit
  --old OLD             path to old training list (default: None)
  --tfr_index TFR_INDEX
                        path to tfrecord index file (default: None)
  --new NEW             path to new training list (default: None)
```

3. Decode the tfrecords to raw image
``` bash
python3 tools/decode.py --help
usage: decode.py [-h] --tfrecords_dir TFRECORDS_DIR --output_dir OUTPUT_DIR
                 --limit LIMIT

decode tfrecord

optional arguments:
  -h, --help            show this help message and exit
  --tfrecords_dir TFRECORDS_DIR
                        path to the output of tfrecords dir path (default:
                        None)
  --output_dir OUTPUT_DIR
                        path to the output of decoded imgs (default: None)
  --limit LIMIT         limit num of decoded samples (default: 10)
```

###  Train

Modified the `DATA_ROOT` and `INDEX_ROOT` in `train.yaml`, `DATA_ROOT` is the parent dir for tfrecord dir,  `INDEX_ROOT` is the parent dir for index file.

```bash
bash local_train.sh
```

### Test

Detail implementations and steps see [Test](https://github.com/Tencent/TFace/tree/master/recognition/test)

### Inference

Detail implementations see [Deploy](https://github.com/Tencent/TFace/tree/master/recognition/deploy)



## Acknowledgement
This repo is modified and adapted on these great repositories, we thank these authors a lot for their greate efforts.
* [cavaface.pytorch](https://github.com/cavalleria/cavaface.pytorch)
* [face.evoLVe.PyTorch](https://github.com/ZhaoJ9014/face.evoLVe.PyTorch) 
* [insightface](https://github.com/deepinsight/insightface)
* [mobile-vision](https://github.com/facebookresearch/mobile-vision)

