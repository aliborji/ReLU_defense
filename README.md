# ReLU_defense
ReLU defense against adversarial attacks


** Work in Progress ... **

Start with intro.ipynb

The master branch works with **PyTorch 1.0.0 or higher.


Manuscript is avaiable in [Arxiv](http://arxiv.org/abs/2004.13013).


## License
This project is released under the [Apache 2.0 license](LICENSE).


## Citation

If you use this code in your research, please cite this project.

```
@article{reluDefense2020,
  title={Harnessing adversarial examples with a surprisingly simple defense},
  author={Borji, Al},
  journal={arXiv preprint arXiv:2004.13013},
  year={2020}
}
```



---------------------
## How to use this defense?

### First, define the srelu function as below:

```
def srelu(input, slope):
    return slope * F.relu(input)
    
class SReLU(nn.Module):
    def __init__(self):
        super().__init__() # init the base class
        
    def forward(self, input, slope):
        return srelu(input, slope)
```
        
        
        
### Second, define your model as (\eg):
```
class NetTest(nn.Module):
    def __init__(self, slope):
        super(NetTest, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        self.slope = slope
        
    def forward(self, x):
        x = srelu(F.max_pool2d(self.conv1(x), 2), self.slope)
        x = srelu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2), self.slope)    
        x = x.view(-1, 320)
        x = srelu(self.fc1(x), self.slope)
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x 
```


### Third, call your model as:
```
model = NetTest(sl).to(device)
```





## Contact

This repo is currently maintained by Ali Borji (aliborji@gmail.com).
