# Cosine Annealing with Warm up for PyTorch

## Example
```
>> model = ...
>> optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-5) # lr is min lr
>> scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=250, T_mult=2, eta_max=0.1, T_up=50)
>> for epoch in range(n_epoch):
>>     train()
>>     valid()
>>     scheduler.step()
```

- case1 : `CosineAnnealingWarmUpRestarts(optimizer, T_0=250, T_mult=1, eta_max=0.1, T_up=50)`
![example1](./src/SGDR2.jpg "example1")
- case2 : `CosineAnnealingWarmUpRestarts(optimizer, T_0=250, T_mult=2, eta_max=0.1, T_up=50)`
![example2](./src/SGDR.jpg "example2")


## 引数
- T_0 : Cosine Annearingのステップ数
- T_multi : ステップの倍率
- eta_max : lrの最大値
- T_up : warmupのイテレーション数 
