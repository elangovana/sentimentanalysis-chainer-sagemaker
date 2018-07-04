# sentimentanalysis-chainer-sagemaker
Sentiment analysis using chainer and sagemaker


# Run
##Train
```bash
python train.py train.csv  --gpu -1  --epoch 1000
```

##Test
```bash
python test.py --gpu -1 --model-setup result/args.json --testset  test.csv
```