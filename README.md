
## DualRec (Disentangling Past-Future Modeling in Sequential Recommendation via Dual Networks)    


## Description   
Sequential recommendation (SR) plays an important role in personalized recommender systems because it captures dynamic and diverse preferences from users’ real-time increasing behaviors. Unlike the standard left-to-right autoregressive training strategy, future data (also available during training) has been used to facilitate model training as it provides richer signals about user’s current interests and can be used to improve the recommendation quality. However, these methods suffer from a severe training-inference gap, i.e., both past and future contexts are modeled by the same encoder when training, while only historical behaviors are available during inference. This discrepancy may lead to potential performance degradation. To alleviate the training-inference gap, we propose a new framework DualRec, which achieves past-future disentanglement and past-future mutual enhancement by a novel dual network. Specifically, a dual network structure is exploited to model the past and future context separately. And a bi-directional knowledge transferring mechanism enhances the knowledge learnt by the dual network. Extensive experiments on four real-world datasets demonstrate the superiority of our approach over baseline methods. Besides, we demonstrate the compatibility of DualRec by instantiating using RNN, Transformer, and filter-MLP as backbones. Further empirical analysis verifies the high utility of modeling future contexts under our DualRec framework.

## How to run   
First, install dependencies   
```bash
# install project   
cd DualRec
pip install -r requirements.txt
```
 Next, run the model with config files for corresponding datasets
 ```bash
python run.py fit --config src/config/{datasets_name}.yaml
 ```
 Then, you can test the model 
  ```bash
python run.py test --config {config_path} --ckpt_path {checkpoint_path}
  ```