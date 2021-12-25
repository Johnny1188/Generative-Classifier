from torch import nn


class ClassifierHeadBlock(nn.Module):

  def __init__(self, layer_shapes=(32,64,32,10), input_dim=1024, verbose=True):
    assert len(layer_shapes) > 1
    super().__init__()

    dense_layers_stack_build = [
      nn.Linear( input_dim, layer_shapes[0]  ),
      nn.ReLU()
    ]
    for dense_layer_i in range(len(layer_shapes)-2):
      dense_layers_stack_build.extend([
        nn.Linear( layer_shapes[dense_layer_i], layer_shapes[dense_layer_i+1]  ),
        nn.ReLU()
      ])
    
    # Add the final layer w/out relu func
    dense_layers_stack_build.append(
      nn.Linear( layer_shapes[-2], layer_shapes[-1] )
    )

    self.dense_layers_stack = nn.Sequential(*dense_layers_stack_build)

    if verbose:
      print(f"-----\n Classifier's dense Head block initialized with layers:\n{self.dense_layers_stack}.\n-----\n")
  
  
  def forward(self, X):
    return( self.dense_layers_stack(X) )

class ClassifierCNNBlock(nn.Module):
  
  def __init__(self, cnn_layers=(16,32,64), input_dims=(28,28,1), verbose=True):
    assert len(input_dims) == 3
    assert len(cnn_layers) > 2
    super().__init__()

    cnn_stack_build = [
      nn.Conv2d( input_dims[-1], cnn_layers[0], kernel_size=3, stride=1, padding=1 ),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2)
    ]

    for cnn_l_id in range(1,len(cnn_layers)):
      cnn_stack_build.extend([
        nn.Conv2d(cnn_layers[cnn_l_id-1],cnn_layers[cnn_l_id],kernel_size=3,stride=1,padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
      ])

    self.cnn_stack = nn.Sequential(*cnn_stack_build)

    if verbose:
      print(f"-----\n Classifier's CNN block initialized with layers:\n{self.cnn_stack}.\n-----\n")


  def forward(self, X):
    return( self.cnn_stack(X) )