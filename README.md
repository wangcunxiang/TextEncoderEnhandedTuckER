
## Commonsense Knowledge Graph Reasoning by Link Prediction: Why and How? 

### Requirements

The codebase is implemented in Python 3.6.6. Required packages are:

```bash
pip3 install numpy
pip3 install pytorch==1.0.1
```

### Running a model

```bash
python main_all.py --dataset ConceptNet --model CNN --num_iterations 20000
python main_all.py --dataset ConceptNet --model LSTM --num_iterations 20000
```
