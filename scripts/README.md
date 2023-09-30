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

## Packaging Experiment for Deployment
Use the `create_ipk.sh` script to create the IPK used to install the experiment on the flatsat and the spacecraft. The IPK is deployed on the spacecraft's Satellite Experimental Processing Platform (SEPP). The SEPP is a powerful ALTERA Cyclone V system-on-chip module with sufficient onboard memory to carry out advanced software and hardware experiments such as running neural network models with TensorFlow Lite. The Altera Cyclone V SX System-on-Chip (SoC) digital core logic device provides an 800 MHz CPU clock and 1GB DDR3 RAM.

To package the experiment for the Engineering Model (EM)'s SEPP, i.e. onboard the flatsat located in the [SMILE Lab](https://www.esa.int/Enabling_Support/Operations/Want_to_SMILE):
```bash
./create_ipk.sh em
```

To package the experiment for the spacecraft's SEPP:
```bash
./create_ipk.sh
```

When installing the IPK on the SEPP the `EXP253_TEST` environment variable can bet set to run the tests immediately after installation:
```bash
EXP253_TEST= opkg install /path/to/package.ipk
```

Alternatively, use the `export` command prior to installing the IPK:
```bash
export EXP253_TEST=
opkg install /path/to/package.ipk
```

Use the `unset` command to remove the environment variable set via the `export` command:
```bash
unset EXP253_TEST
```