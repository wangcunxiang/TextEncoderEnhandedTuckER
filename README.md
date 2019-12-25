
## Commonsense Knowledge Graph Reasoning by Link Prediction: Why and How? 

### Requirements

The codebase is implemented in Python 3.6.6. Required packages are:

    pip3 install numpy
    pip3 install pytorch==1.0.1

### Running a model

This repository contains 

    TEE(CNN + TuckER) in ATOMIC/ConceptNet
    TEE(BiLSTM + TuckER) in ATOMIC/ConceptNet
    TuckER in ATOMIC/ConceptNet

To run the model, execute the following command:

    For TEE(CNN + TuckER) in ATOMIC/ConceptNet:
    
    python main_all.py --model CNN --dataset ATOMIC --num_iterations 10000 --batch_size 64  --edim 300 --rdim 30 --max_length 15
    
    python main_all.py --model CNN --dataset ConceptNet --num_iterations 10000 --batch_size 64  --edim 300 --rdim 50 --max_length 5
    
    For TEE(BiLSTM + TuckER) in ATOMIC/ConceptNet:
    
    python main_all.py --model LSTM --dataset ATOMIC --num_iterations 10000 --batch_size 64  --edim 300 --rdim 30 --max_length 6
    
    python main_all.py --model LSTM --dataset ConceptNet --num_iterations 10000 --batch_size 64  --edim 300 --rdim 50 --max_length 5

```
For TuckER in ATOMIC/ConceptNet:

python main_tucker_pn.py --dataset ATOMIC --num_iterations 10000 --batch_size 64  --edim 300 --rdim 30 

python main_tucker_pn.py --dataset ConceptNet --num_iterations 10000 --batch_size 64  --edim 300 --rdim 50 
```

### Output

For each round of train and evaluation, the output format is

    (Round)
    (time cost in train)
    loss=0.543369487737
    Valid:
    Number of test data points: XXX
    Hits @10: XXX
    Hits @3: XXX
    Hits @1: XXX
    Mean rank: XXX
    Mean reciprocal rank: XXX
    loss=XXX
    (time cost in evaluation)
    Test:
    Number of test data points: XXX
    Hits @10: XXX
    Hits @3: XXX
    Hits @1: XXX
    Mean rank: XXX
    Mean reciprocal rank: XXX
    loss=XXX
    (time cost in evaluation)

