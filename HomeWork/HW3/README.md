# NLP2019
TalTech 2018/2019 NLP Course

## Initial training
~~~
Epoch 10
  Evaluation on train - loss: 0.052825  acc: 97.9200%(9792/10000), f1: 0.915584
  Evaluation on dev - loss: 0.071526  acc: 97.0500%(9705/10000), f1: 0.877948

Epoch 10
  Evaluation on train - loss: 0.023159  acc: 99.2800%(9928/10000), f1: 0.967655
  Evaluation on dev - loss: 0.041919  acc: 98.5100%(9851/10000), f1: 0.928946
  
Epoch 10
  Evaluation on train - loss: 0.030562  acc: 99.1600%(9916/10000), f1: 0.962264
  Evaluation on dev - loss: 0.041759  acc: 98.5600%(9856/10000), f1: 0.931947
  
Epoch 10
  Evaluation on train - loss: 0.027442  acc: 99.2200%(9922/10000), f1: 0.965998
  Evaluation on dev - loss: 0.042289  acc: 98.4800%(9848/10000), f1: 0.931961
  
Epoch 10
  Evaluation on train - loss: 0.024378  acc: 99.2200%(9922/10000), f1: 0.965333
  Evaluation on dev - loss: 0.040406  acc: 98.6200%(9862/10000), f1: 0.935814
  
F1 baseline score out of 5 runs on dev set
Worst: 0.877948
Best:  0.935814
~~~

~~~
class KeywordCnn(nn.Module):
  
  def __init__(self, num_classes, feature_dim, dropout_prob=0.2):
    super(KeywordCnn, self).__init__()
    self.input_bn = nn.BatchNorm1d(feature_dim)

    self.conv1 = nn.Conv1d(feature_dim, 32, kernel_size=10, stride=1)
    self.conv2 = nn.Conv1d(32, 64, kernel_size=8, stride=1)
    self.conv3 = nn.Conv1d(64, 64, kernel_size=5, stride=1)
    
    
    self.conv4 = nn.Conv1d(64, 64, kernel_size=3, stride=1)
    
    
    self.dropout = nn.Dropout(dropout_prob)
    self.fc = nn.Linear(64, num_classes)
    
  def forward(self, x, lengths):
    # Conv1d takes in (batch, channels, seq_len), but raw signal is (batch, seq_len, channels)
    x = x.permute(0, 2, 1).contiguous()
    x = self.input_bn(x)
    x = F.relu(self.conv1(x))
    x = F.max_pool1d(x, 2)
    x = F.relu(self.conv2(x))
    x = F.max_pool1d(x, 2)

    x = F.relu(self.conv3(x))
    x = F.max_pool1d(x, 2)

    
    
    x = F.relu(self.conv4(x))
    # Global max pooling
    x = F.max_pool1d(x, x.size(2))
    x = x.view(-1, 64)
    x = self.dropout(x) 
    logit = self.fc(x)
    return logit
~~~    

~~~    
Epoch 10
Evaluation on train - loss: 0.016763  acc: 99.5500%(9955/10000), f1: 0.980132
Evaluation on dev - loss: 0.036148  acc: 98.8100%(9881/10000), f1: 0.945438
~~~    


~~~    
    self.conv1 = nn.Conv1d(feature_dim, 32, kernel_size=10, stride=1)
    self.conv2 = nn.Conv1d(32, 64, kernel_size=8, stride=1)
    self.conv3 = nn.Conv1d(64, 64, kernel_size=6, stride=1)
    self.conv4 = nn.Conv1d(64, 64, kernel_size=4, stride=1)

Epoch 10
  Evaluation on train - loss: 0.015516  acc: 99.6200%(9962/10000), f1: 0.983096
  Evaluation on dev - loss: 0.039075  acc: 98.7800%(9878/10000), f1: 0.942723
~~~    

~~~    
    self.conv1 = nn.Conv1d(feature_dim, 32, kernel_size=20, stride=1)
    self.conv2 = nn.Conv1d(32, 64, kernel_size=16, stride=1)
    self.conv3 = nn.Conv1d(64, 64, kernel_size=12, stride=1)
    self.conv4 = nn.Conv1d(64, 64, kernel_size=6, stride=1)
    
Epoch 10
  Evaluation on train - loss: 0.016444  acc: 99.5400%(9954/10000), f1: 0.979842
  Evaluation on dev - loss: 0.031604  acc: 99.0500%(9905/10000), f1: 0.956641
~~~    

~~~    
    self.conv1 = nn.Conv1d(feature_dim, 32, kernel_size=18, stride=1)
    self.conv2 = nn.Conv1d(32, 64, kernel_size=16, stride=1)
    self.conv3 = nn.Conv1d(64, 64, kernel_size=12, stride=1)
    self.conv4 = nn.Conv1d(64, 64, kernel_size=6, stride=1)

Epoch 20
  Evaluation on train - loss: 0.005657  acc: 99.8600%(9986/10000), f1: 0.993865
  Evaluation on dev - loss: 0.029047  acc: 99.1200%(9912/10000), f1: 0.959891
~~~    

~~~    
    self.conv1 = nn.Conv1d(feature_dim, 32, kernel_size=18, stride=1)
    self.conv2 = nn.Conv1d(32, 64, kernel_size=14, stride=1)
    self.conv3 = nn.Conv1d(64, 64, kernel_size=10, stride=1)
    self.conv4 = nn.Conv1d(64, 64, kernel_size=6, stride=1)
    
Epoch 100
  Evaluation on train - loss: 0.011924  acc: 99.7700%(9977/10000), f1: 0.989987
  Evaluation on dev - loss: 0.077070  acc: 99.0000%(9900/10000), f1: 0.954710
~~~
