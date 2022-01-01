import torch
from torch import nn


class GeneratorHeadBlock(nn.Module):

    def __init__(self,
               classifier_cnn_block_in_layer_shapes=(512,458),
               classifier_prediction_in_layer_shapes=(10,64),
               num_of_classes=10,
               main_layer_shapes=(522,512,1024)):
        assert len(classifier_cnn_block_in_layer_shapes) > 0
        assert len(classifier_prediction_in_layer_shapes) > 0
        assert main_layer_shapes[0] == classifier_cnn_block_in_layer_shapes[-1] + classifier_prediction_in_layer_shapes[-1]
        super().__init__()

        self.num_of_classes = num_of_classes

        self.dense_layers_stack_dict = {
            "in_classifier_cnn_block": [],
            "in_classifier_prediction": [],
            "in_combined_main_stack": []
        }

        # Layers processing the input coming from the Classifier's CNN block
        for dense_layer_i in range(1,len(classifier_cnn_block_in_layer_shapes)):
            self.dense_layers_stack_dict["in_classifier_cnn_block"].extend([
                nn.Linear( classifier_cnn_block_in_layer_shapes[dense_layer_i-1], classifier_cnn_block_in_layer_shapes[dense_layer_i] ),
                nn.ReLU()
            ])
        # Layers processing the input coming from the Classifier's prediction block
        for dense_layer_i in range(1,len(classifier_prediction_in_layer_shapes)):
            self.dense_layers_stack_dict["in_classifier_prediction"].extend([
                nn.Linear( classifier_prediction_in_layer_shapes[dense_layer_i-1], classifier_prediction_in_layer_shapes[dense_layer_i] ),
                nn.ReLU()
            ])
        # Layers combining the two input streams
        for dense_layer_i in range(1,len(main_layer_shapes)):
            self.dense_layers_stack_dict["in_combined_main_stack"].extend([
                nn.Linear( main_layer_shapes[dense_layer_i-1], main_layer_shapes[dense_layer_i]  ),
                nn.ReLU()
            ])

        self.dense_layers_stack_dict["in_classifier_cnn_block"] = nn.Sequential(*self.dense_layers_stack_dict["in_classifier_cnn_block"])
        self.dense_layers_stack_dict["in_classifier_prediction"] = nn.Sequential(*self.dense_layers_stack_dict["in_classifier_prediction"])
        self.dense_layers_stack_dict["in_combined_main_stack"] = nn.Sequential(*self.dense_layers_stack_dict["in_combined_main_stack"])

    def forward(self, X_from_cnn_block, X_from_prediction):
        processed_cnn_block_input = self.dense_layers_stack_dict["in_classifier_cnn_block"](X_from_cnn_block)
        processed_prediction_input = self.dense_layers_stack_dict["in_classifier_prediction"](X_from_prediction)
        return(self.dense_layers_stack_dict["in_combined_main_stack"](torch.cat((processed_cnn_block_input, processed_prediction_input),1)))

class GeneratorCNNBlock(nn.Module):

    def __init__(self, cnn_transpose_layers=(32,16,1), input_dims=(64,4,4)):
        assert len(cnn_transpose_layers) > 1
        assert len(input_dims) > 1
        super().__init__()

        cnn_trans_layers_build = []

        # First cnn layer receiving the input image
        cnn_trans_layers_build.extend([
            nn.ConvTranspose2d(input_dims[0], cnn_transpose_layers[0], kernel_size=2, stride=2),
            nn.LeakyReLU(negative_slope=0.02),
            nn.ConvTranspose2d(cnn_transpose_layers[0], cnn_transpose_layers[1], kernel_size=3, stride=2),
            nn.LeakyReLU(negative_slope=0.02),
        ])

        # Subsequent transpose cnn layers (input shape=output shape of previous layer)
        for cnn_l_id in range(2, len(cnn_transpose_layers)-1):
            cnn_trans_layers_build.extend([
                nn.ConvTranspose2d(cnn_transpose_layers[cnn_l_id-1], cnn_transpose_layers[cnn_l_id], kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(negative_slope=0.02)
            ])
        # Last sigmoid because pixel values between 1 and 0
        cnn_trans_layers_build.extend([
            nn.ConvTranspose2d(cnn_transpose_layers[-2], cnn_transpose_layers[-1], kernel_size=2, stride=1, padding=1),
            nn.Sigmoid()
        ])
        
        self.cnn_trans_stack = nn.Sequential(*cnn_trans_layers_build)

    def forward(self,X):
        return(self.cnn_trans_stack(X))
