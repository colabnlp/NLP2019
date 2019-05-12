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

~~~
    self.conv1 = nn.Conv1d(feature_dim, 32, kernel_size=16, stride=1)
    self.conv2 = nn.Conv1d(32, 64, kernel_size=14, stride=1)
    self.conv3 = nn.Conv1d(64, 64, kernel_size=12, stride=1)
    self.conv4 = nn.Conv1d(64, 64, kernel_size=6, stride=1)

Epoch 10
  Evaluation on train - loss: 0.011490  acc: 99.7200%(9972/10000), f1: 0.987794
  Evaluation on dev - loss: 0.026598  acc: 99.2700%(9927/10000), f1: 0.966953
~~~

Attempt at label smoothing - not working
RuntimeError: The size of tensor a (32) must match the size of tensor b (3) at non-singleton dimension 1
~~~
def train(model, num_epochs, train_iter, dev_iter, device, log_interval=10, label_smoothing=0.1):

  assert (label_smoothing >= 0.0 and label_smoothing <= 1.0)
  # Each non-target class gets a target probability of label_smoothing / 2.0
  non_target_prob = label_smoothing / 2.0
  
  optimizer = torch.optim.Adam(model.parameters())

  steps = 0
  best_acc = 0
  last_step = 0
  
  criterion = nn.KLDivLoss(reduction='sum')
  
  for epoch in range(1, num_epochs+1):
    print("Epoch %d" % epoch)
    for batch in train_iter:
      # set training mode
      model.train()
      features, target = batch.features, batch.labels
     
      target_double = target.double()
      #tensor([1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], device='cuda:0')
      target_higher = torch.add(target_double, -label_smoothing)
      target_lower = torch.add(target_double, non_target_prob)
      
      target_smoothed = torch.where( (target > 0), target_higher, target_lower)
      
      #print(target)
      #raise Exception('debug')
      
      optimizer.zero_grad()
      logit = model(features[0], features[1])

      
      log_probabilities = F.log_softmax(logit)
      loss = criterion(log_probabilities, target_smoothed).mean()
      
      #loss = F.cross_entropy(logit, target)

      loss.backward()
      optimizer.step()

      steps += 1

    train_acc = evaluate("train", train_iter, model)                
    dev_acc = evaluate("dev", dev_iter, model)

~~~

~~~
import sklearn.metrics
import torch.optim



def train(model, num_epochs, train_iter, dev_iter, device, log_interval=10, label_smoothing=0.0):

  assert (label_smoothing >= 0.0 and label_smoothing <= 1.0)
  # Each non-target class gets a target probability of label_smoothing / 2.0
  non_target_prob = label_smoothing / 2.0
  

  steps = 0
  best_acc = 0
  last_step = 0
  
  learning_rate = 0.1
  momentum = 0.9

  #optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)
  optimizer = torch.optim.Adam(model.parameters(), amsgrad=True)
  #optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate,momentum=momentum, weight_decay=0.0001)
  #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

  
  best_f1_train = 0
  best_f1_dev = 0
  best_f1_test = 0

  best_f1_train_epoch = 0
  best_f1_dev_epoch = 0
  best_f1_test_epoch = 0


  criterion = nn.KLDivLoss(reduction='sum', size_average=False)
  
  for epoch in range(1, num_epochs+1):
    print("Epoch %d" % epoch)
    #scheduler.step(epoch)
    for batch in train_iter:
      # set training mode
      model.train()
      features, target = batch.features, batch.labels
     
      if label_smoothing > 0.001:
        target_double = target.double()
        #tensor([1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], device='cuda:0')
        target_higher = torch.add(target_double, -label_smoothing)
        target_lower = torch.add(target_double, non_target_prob)
      
        target_smoothed = torch.where( (target > 0), target_higher, target_lower)
      
        #print(target)
        #raise Exception('debug')
      
        optimizer.zero_grad()
        logit = model(features[0], features[1])

      
        log_probabilities = F.log_softmax(logit, dim=1)
        asi = (log_probabilities*target_smoothed).sum(dim=1)
        print(asi)
      
        print(0)

        print(log_probabilities)
        print(target_smoothed)
      
        print(F.cross_entropy(logit, target))
      
      
        loss = criterion(log_probabilities, target_smoothed).mean()
      
        print(1)
      else:
        optimizer.zero_grad()
        logit = model(features[0], features[1])
        loss = F.cross_entropy(logit, target)

      loss.backward()
      optimizer.step()
      
      steps += 1
     
    (_,_,f1_train_acc) = evaluate("train", train_iter, model)                
    (_,_,f1_dev_acc) = evaluate("dev", dev_iter, model)
    (_,_,f1_test_acc) = evaluate("test", test_iter, model)

    print(f1_train_acc, f1_dev_acc, f1_test_acc)

    if f1_train_acc > best_f1_train:
      best_f1_train=f1_train_acc
      best_f1_train_epoch = epoch

    if f1_dev_acc > best_f1_dev:
      best_f1_dev=f1_dev_acc
      best_f1_dev_epoch = epoch

    if f1_dev_acc > best_f1_test:
      best_f1_test=f1_train_acc
      best_f1_test_epoch = epoch

  print("train", best_f1_train, best_f1_train_epoch)
  print("dev", best_f1_dev, best_f1_dev_epoch)
  print("test", best_f1_test, best_f1_test_epoch)




def evaluate(name, data_iter, model):
  # set evaluation mode (turns off dropout)
  model.eval()
  corrects, avg_loss = 0, 0
  y_true = torch.tensor([]).long()
  y_pred = torch.tensor([]).long()
  for batch in data_iter:
    features, target = batch.features, batch.labels

    logit = model(features[0], features[1])
    loss = F.cross_entropy(logit, target, reduction='sum')

    avg_loss += loss.item()
    batch_predictions = torch.max(logit, 1)[1].view(target.size()).data
    
    corrects += (batch_predictions == target.data).sum()
    
    y_pred = torch.cat((y_pred, batch_predictions.detach().cpu()))
    y_true = torch.cat((y_true, target.detach().cpu()))

  size = len(data_iter.dataset)
  avg_loss /= size
  accuracy = 100.0 * float(corrects)/size
  
  f1 = sklearn.metrics.f1_score(y_true, y_pred)
  
  print('  Evaluation on {} - loss: {:.6f}  acc: {:.4f}%({}/{}), f1: {:4f}'.format(
      name,
      avg_loss, 
      accuracy, 
      corrects, 
      size,
      f1))
  return accuracy, avg_loss, f1




  class KeywordCnn(nn.Module):
  
  def __init__(self, num_classes, feature_dim, dropout_prob=0.2):
    super(KeywordCnn, self).__init__()
    self.input_bn = nn.BatchNorm1d(feature_dim)
    self.conv1 = nn.Conv1d(feature_dim, 32, kernel_size=16, stride=1)
    self.conv2 = nn.Conv1d(32, 64, kernel_size=14, stride=1)
    self.conv3 = nn.Conv1d(64, 64, kernel_size=12, stride=1)
    self.conv4 = nn.Conv1d(64, 64, kernel_size=6, stride=1)
    
    
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

# Let's train the baseline model:  
model = KeywordCnn(3, 13).to(device)
train(model, 1000, train_iter, dev_iter, device=device)

Epoch 1000
  Evaluation on train - loss: 0.000001  acc: 100.0000%(10000/10000), f1: 1.000000
  Evaluation on dev - loss: 0.083059  acc: 99.4900%(9949/10000), f1: 0.976552
  Evaluation on test - loss: 0.078971  acc: 99.5100%(9951/10000), f1: 0.977778
1.0 0.976551724137931 0.9777777777777777
train 1.0 23
dev 0.9788602941176472 945
test 0.9882352941176471 13

Epoch 945
  Evaluation on train - loss: 0.000004  acc: 100.0000%(10000/10000), f1: 1.000000
  Evaluation on dev - loss: 0.080962  acc: 99.5400%(9954/10000), f1: 0.978860
  Evaluation on test - loss: 0.067596  acc: 99.5700%(9957/10000), f1: 0.980499
1.0 0.9788602941176472 0.9804988662131519
Epoch 13
  Evaluation on train - loss: 0.009493  acc: 99.7300%(9973/10000), f1: 0.988235
  Evaluation on dev - loss: 0.025310  acc: 99.2100%(9921/10000), f1: 0.964075
  Evaluation on test - loss: 0.028909  acc: 99.1300%(9913/10000), f1: 0.960969
0.9882352941176471 0.964074579354252 0.9609690444145357
Epoch 23
  Evaluation on train - loss: 0.000929  acc: 100.0000%(10000/10000), f1: 1.000000
  Evaluation on dev - loss: 0.041797  acc: 99.1300%(9913/10000), f1: 0.959815
  Evaluation on test - loss: 0.043699  acc: 99.2100%(9921/10000), f1: 0.964042
1.0 0.9598152424942262 0.9640418752844787
~~~