<h1 align="center">Handwritten Digit Recognition</h1>

<p align="center">
    A machine learning model that recognizes handwritten digits based on my
    <a href="https://github.com/niklashenning/hwd-1000-dataset">HwD-1000 dataset</a>.
</p>

<p align="center">
    <img src="https://github.com/user-attachments/assets/3ca61c98-4403-4f69-b34e-17dbe8139fc3" width="250">
</p>


## About
This basic machine learning model is trained on the [HwD-1000 dataset](https://github.com/niklashenning/hwd-1000-dataset)
and is able to accurately classify and recognize digits from images.<br>
It utilizes PyTorch's neural networks and Tensor libraries for training the model, pandas for loading
and handling the dataset, Pillow for loading and converting the images, and Matplotlib
for visualizing the training results.


## Dataset
The dataset contains 1000 images of single digits (0-9) that have been manually drawn on a white
28x28 px background with a black pen in varying widths and styles.

<img src="https://github.com/user-attachments/assets/6b84dd91-b959-472a-a3b6-e327c1826a3d">


## Results
The neural network was trained for 50 epochs with a learning rate of 0.001 using AdamW
as the optimizer and CrossEntropyLoss as the criterion. Training was done on 80%
of the dataset while the remaining 20% were used to validate the results.

The model achieved an accuracy of 99.50% on the validation data.

```
Test Accuracy: 99.50% (199/200)
```

| Epoch | Loss     |
|-------|----------|
| 1     | 1.506903 |
| 10    | 0.047416 |
| 20    | 0.011299 |
| 30    | 0.006869 |
| 40    | 0.002777 |
| 50    | 0.001392 |

<img src="https://github.com/user-attachments/assets/70aec14d-2f0c-4b1f-a39c-050bc1db0cbb" width="500">


## License
This software is licensed under the [MIT license](LICENSE).
