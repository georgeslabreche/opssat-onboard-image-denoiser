## Prepare Unnoised Images
Split the training images into patches:
```bash
python batch_splitter.py -r 224 -s 56
```

Split the validation images into patches:
```bash
python batch_splitter.py -r 224 -s 56 -v
```

## Prepare Noised Images
Add noise to the training images and then split them into patches:
```bash
bash batch_noiser.sh ../data/opssat/earth
python batch_splitter.py -s 56 -n
```

Add noise to the validation images and then split them into patches (is not needed?):
```bash
bash batch_noiser.sh ../data/opssat/validate
python batch_splitter.py -s 56 -v -n
```

Note that when invoking `batch_splitter.py` on noised images we do not specify resizing the input image (i.e. `-r 224`). This is because `batch_noiser.sh` already does that resizing.

## Batch Training
```bash
python batch_train_autoencoders.py
```
