# BPCIN

Official Repository for "Bigraph Proto-calibration Cross-modal Inference Network for Few-shot Visual Question Answering".

## Installation

```bash
conda create -n bpcin python=3.9
conda activate bpcin

pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

python -m pip install 'https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.5.0/en_core_web_lg-3.5.0-py3-none-any.whl'
python -m pip install 'git+https://github.com/openai/CLIP.git'

pip install -r requirements.txt
```

## Setup

### Environment

Please modify the source code of CLIP in the following ways (clip/model.py).

In class **ModifiedResNet** (function **forward**)，please add and modify following lines.

```python
def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        last_sec_layer = x		# added
        x = self.layer4(x)
        last_layer = x			# added
        x = self.attnpool(x)

        return x, last_sec_layer, last_layer		# modified
```

### Datasets

You may download datasets here.

### Checkpoints

You may download checkpoints here.

### Getting Started

#### Train

```bash
python main.py \
  --mode train \
  --device cuda:0 \
  --config ./config/5way_5shot_clip_coco_qa.py \
  --dataset_root /path/to/COCO_QA
```

#### Test

```bash
python main.py \
  --mode eval \
  --device cuda:0 \
  --config ./config/5way_5shot_clip_coco_qa.py \
  --dataset_root /path/to/COCO_QA \
  --checkpoint_dir ./fpait_checkpoints
```

