# kaggle_m5_forecasting

Solution for Kaggle M5 Forecasting Competition (https://www.kaggle.com/c/m5-forecasting-accuracy/overview)

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
