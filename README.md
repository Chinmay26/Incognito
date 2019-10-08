# Incognito
This project removes your background by optimizing segmentation models for CPU inference. It applies deep learning optimizations & engineering optimizations to improve CPU inference frame rate from 3 FPS to 15 FPS.

## Results

### Visual Results

### Optimization Results
|          Optimization                      | CPU Inference time    |        FPS      |       Model Size      |       IOU      |
| ------------------------------------------ |:---------------------:| ----------------|-----------------------|----------------|
| ResNet Model (256*256)                     |  ~265 ms              |        3-4      |         98.2 MB       |   0.95592      |
| Architecture Optimization                  |  ~183 ms              |        5-6      |         81.2 MB       |   0.96079      |
| Deep Learning Optimization                 |  ~179 ms              |        5-6      |         10.6 MB       |   0.96079      |
| Engineering Optimization (128 * 128)       |  ~156 ms              |       15-17     |         10.6 MB       |   0.94648      |

### Model files
The optimized and compressed models can be found under following directory
```
/models/optimized_models
```


## Requirements
The package as well as the necessary requirements can be installed by running `make` or via
```
python setup.py install
```

## Presentation Slides
Further details regarding the motivation, methods and results of  different optimization
techniques can be found in my presentation 
<a href="https://docs.google.com/presentation/d/1kzGghFEe5G4dSuMc2pVOZP2ChKCjWENIX7n6D-AA1y4/edit?usp=sharing" target="_blank">here</a>.

## License

[MIT License](LICENSE)

Copyright (c) 2019 Chinmay Naik
