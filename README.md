# Sentimentanalysis using chainer and Sagemaker
Sentiment analysis using chainer and sagemaker

## Data prep
This is to shuffle split the data file into train and test set.

**Note:** This may be a slow based on the size of the input file.
### Shuffle
```bash
export PYTHONPATH=./custom_chainer
export inputyelpreviewfile=~/data/yelp_review.csv ## Specfy the path to the yelp review file
export outputdatadir=~/data ##Specify the output directory to place the 2 output files
python ./custom_chainer/dataprep/Main.py $inputyelpreviewfile $outputdatadir shuffle --first-file-name yelp_review_train.shuffled.csv --second-file-name yelp_review_test.shuffled.csv
```

### Convert to blazing text format 
**Note: Only required if using blazing text classification**
```bash
python custom_chainer/datasetyelp/main_blazingtextformatter.py ./tests/data/sample_train.csv /tmp/btext.txt

```


## How to run
### 1. Train locally
```bash
python custom_chainer/main_full_cycle.py  --dataset-type yelp --traindata tests/data/sample_train.csv   -g -1  --epoch 100 --out result
```

```bash
python custom_chainer/main_full_cycle.py  --dataset-type movie --traindata tests/data/movies_sample_dataset.csv   -g -1  --epoch 100 --out result
```

### 2. Test locally
```bash
python custom_chainer/main_predict.py --gpu -1 --model-setup result/args.json --testset  custom_chainer/tests/data/test.csv
```

### 3. To use sagemaker
Use the sentiment_analysis.ipynb notebook


