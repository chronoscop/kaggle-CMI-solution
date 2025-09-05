# kaggle-CMI-Public/Private-15-29-solution
Training script for the [CMI - Detect Behavior with Sensor Data competition](https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data)

A detailed description can be found in our [kaggle writeup](https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data/writeups/29th-place-solution-handedness-augmentation-trick)

---

## Requirements

To run the code, install the required packages:

```bash
pip install -r requirements.txt
```

------

## Usage

1. Download the dataset from [Kaggle](https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data/data) and put the data in the `./data` fold

   Place it under the `./data` folder, keeping the original structure:

   ```plaintext
   ./data/
   ├── kaggle_evaluation
   ├── test.csv
   ├── test_demographics.csv
   ├── train.csv
   └── train_demographics.csv
   ```

2. run the script in the `./script` fold

------

## Scripts

- [`cmi-1d-cnn_I2nfinit3y.py`](.\script\cmi-1d-cnn_I2nfinit3y.py) - I2nfinit3y's 1D CNN training script with full features model.
- [`cmi-1d-cnn-imu-only-I2nfinit3y.py`](.\script\cmi-1d-cnn-imu-only-I2nfinit3y.py) - I2nfinit3y's 1D CNN training script with IMU only features model.
- [`XXX-CMI-ver27-top.py`](.\script\XXX-CMI-ver27-top.py) - XXX's training script with full features model and different feature engineering.
- [`XXX-CMI-ver37-top.py`](.\script\XXX-CMI-ver37-top.py) - XXX's training script with full features model and different feature engineering.
- [`XXX-CMI-ver40-top.py`](.\script\XXX-CMI-ver40-top.py) - XXX's training script with full features model and different feature engineering

#### Additional scripts

please check the `./others` fold

------

## Result

| script                            | OOF CV | Public LB | Private LB |
| --------------------------------- | ------ | --------- | ---------- |
| cmi-1d-cnn_I2nfinit3y.py          | 0.8800 | 0.861     | 0.836      |
| cmi-1d-cnn-imu-only-I2nfinit3y.py | 0.8336 | 0.835     | 0.818      |
| XXX-CMI-ver37-top.py              | 0.8785 | 0.853     | 0.848      |
| XXX-CMI-ver40-top.py              | 0.8758 | 0.855     | 0.843      |

