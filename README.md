# sentimentanalysis-chainer-sagemaker
Sentiment analysis using chainer and sagemaker


# Run
##Train locally
```bash
python custom_chainer/train.py  custom_chainer/tests/data/sample_train.csv   -g -1  --epoch 100 --out result
```

##Test locally
```bash
python test.py --gpu -1 --model-setup result/args.json --testset  test.csv
```