# Dog Breed Classifier with SageMaker

This project focuses on building and deploying a ML model to classify dog breeds using Amazon SageMaker. The model is trained on a dataset of labeled dog images, where each label corresponds to a specific breed.

## Project Set Up and Installation
A local development environment was configured using a Docker container with Amazon SageMaker client libraries, leveraging an official image downloaded from AWS. This approach enabled seamless development and execution of code within a Jupyter Notebook environment, ensuring compatibility with SageMaker's APIs and frameworks.

Using a local container allowed for efficient resource utilization by minimizing reliance on cloud-based instances during the initial stages of development. This setup also provided the flexibility to iterate quickly on the code and test functionality locally before deploying models to SageMaker. As a result, more time was spent refining the project rather than managing cloud resources.


## Dataset
Training / Validateion / Test dataset contained multiple images of 133 breeds and was downloaded from AWS by
```
wget https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip
```
It was downloded to local machine and later uploaded using command line aws and stored in S3
```
aws s3 sync
```
<img width="1110" alt="Screen Shot 2024-12-07 at 22 34 32" src="https://github.com/user-attachments/assets/67278ad8-5b5e-4dc4-814f-b2ef0f9ac136">



## Hyperparameter Tuning
For this experiment, a pre-trained ResNet-18 model was chosen as the base architecture. 
Hyperparameter optimization was performed to find the best configuration for training the model. The following parameters and their ranges were used for the search:

__Learning Rate:__
Range: 0.0001 to 0.01
Affects how quickly the model updates weights during training. Smaller values ensure stability, while larger values speed up convergence but may overshoot optimal points.

__Batch Size:__
Values: 16, 32, 64
Determines the number of samples processed before the model updates its weights. Smaller sizes increase iteration frequency but require more training time.

__Number of Epochs:__
Values: 5, 10, 20
The number of complete passes through the dataset. This was balanced to prevent overfitting while allowing sufficient training time.
Optimizer:

__Choices: Adam, SGD__
Adam is known for its adaptive learning rate, while SGD often converges to better optima with proper tuning.

Completed Hyperparameter Tuning Jobs:
- pytorch-training-230303-0254
- pytorch-training-230228-1100
- pytorch-training-230227-1323
```
Best hyperparameters
            job_name                  batch_size    leqrning rate           FinalObjectiveValue
0  pytorch-training-230303-0254       "32"          0.001378                149.0
1  pytorch-training-230228-1100       "32"          0.013671                121.0
2  pytorch-training-230227-1323       "32"          0.082775                 55.0
```

  <img width="702" alt="Screen Shot 2024-12-07 at 22 06 21" src="https://github.com/user-attachments/assets/ca955b04-734b-4aca-a411-5fb8f3a4fb18">
                    

## Debugging and Profiling

Debugging
1. Enabling Debugging Hooks:
I enabled SageMaker Debugger by specifying debugger_hook_config in the SageMaker Estimator. This automatically captured training metrics such as loss and gradients.
2. SageMaker Debugger tracked metrics such as loss, weights, and gradients during training.
Using SageMaker Debugger’s smdebug library, I accessed training data to visualize and analyze the training process


```
from sagemaker.debugger import DebuggerHookConfig

debugger_hook_config = DebuggerHookConfig(
    s3_output_path=f"s3://{BUCKET}/debug-output"  # Specify output location for debugging artifacts
)
```
There were some issues found, some images were corrupted. The solution was to ignore Truncated Images: Modify the dataset loader to ignore corrupted images by wrapping the loading logic in a try-except.  I set ImageFile.LOAD_TRUNCATED_IMAGES to True globally. This allows Pillow to open truncated images without raising an error
```
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
```

Profiling
1. Enabling Profiling Tools:
SageMaker Profiler was configured to monitor system resource usage (CPU, GPU, memory, I/O, etc.) during training. Here are the first roecored rows:

```
                     timestamp  timestamp_us        value                      system_metric          dimension  nodeID     type 
0   2024-12-08T05:48:39:508406      39508406         6.18                  MemoryUsedPercent                     algo-1   memory  
1   2024-12-08T05:48:40:008990      40008990         0.00                               IOPS                     algo-1      i/o  
2   2024-12-08T05:48:40:008990      40008990         0.00     ReadThroughputInBytesPerSecond                     algo-1      i/o  
3   2024-12-08T05:48:40:008990      40008990         0.00    WriteThroughputInBytesPerSecond                     algo-1      i/o  
4   2024-12-08T05:48:40:009090      40009090        98.00                               cpu0     CPUUtilization  algo-1      cpu  
5   2024-12-08T05:48:40:009090      40009090        14.00                               cpu0  I/OWaitPercentage  algo-1      cpu  
6   2024-12-08T05:48:40:009090      40009090        89.58                               cpu1     CPUUtilization  algo-1      cpu  
7   2024-12-08T05:48:40:009090      40009090         4.17                               cpu1  I/OWaitPercentage  algo-1      cpu  
8   2024-12-08T05:48:40:009163      40009163         6.75                  MemoryUsedPercent                     algo-1   memory  
9   2024-12-08T05:48:40:013693      40013693         0.00              ReceiveBytesPerSecond          Algorithm  algo-1  network  
10  2024-12-08T05:48:40:013693      40013693         0.00             TransmitBytesPerSecond          Algorithm  algo-1  network  
11  2024-12-08T05:48:40:013693      40013693  66775126.66              ReceiveBytesPerSecond           Platform  algo-1  network  
12  2024-12-08T05:48:40:013693      40013693     70864.81             TransmitBytesPerSecond           Platform  algo-1  network  
13  2024-12-08T05:48:40:508502      40508502         7.32                  MemoryUsedPercent                     algo-1   memory  
14  2024-12-08T05:48:40:508563      40508563        98.04                               cpu0     CPUUtilization  algo-1      cpu  
15  2024-12-08T05:48:40:508563      40508563         3.92                               cpu0  I/OWaitPercentage  algo-1      cpu  
16  2024-12-08T05:48:40:508563      40508563       100.00                               cpu1     CPUUtilization  algo-1      cpu  
17  2024-12-08T05:48:40:508563      40508563         6.00                               cpu1  I/OWaitPercentage  algo-1      cpu  
18  2024-12-08T05:48:40:508622      40508622      5357.94                               IOPS                     algo-1      i/o  
19  2024-12-08T05:48:40:508622      40508622   8970700.56     ReadThroughputInBytesPerSecond                     algo-1      i/o  
```


### Results
```
Epoch 1/5
Training Loss: 2.686378387743206
Test Loss: 2.2603519978346647
Accuracy: 36.287425149700596%
Epoch 2/5
Training Loss: 1.368188729696867
Test Loss: 2.225697808795505
Accuracy: 41.19760479041916%
Epoch 3/5
Training Loss: 0.8129068861167397
Test Loss: 1.7059940966191116
Accuracy: 50.778443113772454%
Epoch 4/5
Training Loss: 0.5381081418700195
Test Loss: 1.6437258786625333
Accuracy: 55.5688622754491%
Epoch 5/5
Training Loss: 0.341343711330845
Test Loss: 1.505983540305385
Accuracy: 60.119760479041915%
```

<img width="666" alt="Screen Shot 2024-12-07 at 22 01 40" src="https://github.com/user-attachments/assets/2cbd8c46-bdf1-4f8b-af20-b698ff10a794">


## Model Deployment

In this project, the model was deployed to an Amazon SageMaker endpoint using a PyTorch-based model that classifies dog breeds. After training the model, the deployment process allows you to invoke the model through an HTTP API endpoint. This endpoint is hosted on an Amazon SageMaker instance and can accept image inputs to classify the dog breed.
input - image 

![French_bulldog_04821](https://github.com/user-attachments/assets/9f8e8867-cfcf-4055-8fdf-50f13ffdc014)

```
{
  "image": "bytes"
}
{
  "predicted_label": "069",
  "class_name": "French_bulldog"
}
```

## Standout Suggestions
**TODO (Optional):** This is where you can provide information about any standout suggestions that you have attempted.

