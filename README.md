# Sentimentanalysis using chainer and Sagemaker
Sentiment analysis using chainer and sagemaker

## Data prep
This is to shuffle split the data file into train and test set.

**Note:** This may be a slow based on the size of the input file.

```bash
export PYTHONPATH=./custom_chainer
export inputyelpreviewfile=~/data/yelp_review.csv ## Specfy the path to the yelp review file
export outputdatadir=~/data ##Specify the output directory to place the 2 output files
python ./custom_chainer/dataprep/splitter.py $inputyelpreviewfile $outputdatadir shuffle --first-file-name yelp_review_train.shuffled.csv --second-file-name yelp_review_test.shuffled.csv
```
## How to run
### 1. Train locally
```bash
python custom_chainer/main_full_cycle.py  --traindata custom_chainer/tests/data/sample_train.csv   -g -1  --epoch 100 --out result
```

### 2. Test locally
```bash
python custom_chainer/main_predict.py --gpu -1 --model-setup result/args.json --testset  custom_chainer/tests/data/test.csv
```

### 3. To use sagemaker
Use the sentiment_analysis.ipynb notebook
