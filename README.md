# kaggle_m5_forecasting

Solution for Kaggle M5 Forecasting Competition (https://www.kaggle.com/c/m5-forecasting-accuracy/overview)

## PB LeaderBoard History

2020-04-02 0.64561 (first submission)

2020-04-04 0.63581

2020-04-04 0.55002 (thanks to dark magic https://www.kaggle.com/kyakovlev/m5-dark-magic)

2020-04-05 0.53538

2020-04-10 0.51792

2020-04-11 0.50514

2020-04-11 0.48833 (thanks to iterative prediction https://www.kaggle.com/kneroma/m5-first-public-notebook-under-0-50)

2020-04-12 0.48273

## Ponchi

![Alt text](appendix/ponchi.png?raw=true "Ponchi")

## Step1. Set up environments with pipenv

```bash
pipenv install --dev --skip-lock
```

## Step2. Download data

Put them in ./m5-forecasting-accuracy directory.

## Step3. Start up luigi / mlflow server (in other terminal windows)

```bash
luigid
```

```bash
mlflow ui
```

## Step4. Create submission file

```bash
pipenv run python main.py m5.LGBMSubmission
```
