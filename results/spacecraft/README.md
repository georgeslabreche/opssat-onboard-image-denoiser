# Spacecraft Results

Experiment results on the FM spacecraft.

- **[wgan_fpn50_1695963889066](./wgan_fpn50_1695963889066):** WGAN runs for fixed-pattern noise (FPN) 50.
- **[wgan_fpn50_1695964476824](./wgan_fpn50_1695964476824):** WGAN runs for FPN 50.
- **[wgan_fpn50_1697455926224](./wgan_fpn50_1697455926224):** WGAN runs for FPN 50.

To run the python scripts that calculate the image similarity metrics and generates a csv with its plot:

## Example usage

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r ../requirements.txt
python3 calculate_metrics.py --csv_input_file ./csv/results_classification-WGAN-FPN-50-short.csv --csv_output_file ./csv/results_classification-WGAN-FPN-50-metrics.csv --original_folder ./images/WGAN/FPN-50/ --denoised_folder ./images/WGAN/FPN-50/
```
