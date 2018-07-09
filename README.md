# sentimentanalysis-chainer-sagemaker
Sentiment analysis using chainer and sagemaker


# Run
## 1. Train locally
```bash
python -m custom_chainer.main_full_cycle  --traindata tests/data/sample_train.csv   -g -1  --epoch 100 --out result
```

## 2. Test locally
```bash
python -m custom_chainer.main_predict --gpu -1 --model-setup result/args.json --testset  tests/data/test.csv
```