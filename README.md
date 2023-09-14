# Dockerized LSTM Toxic Comment Classification Model

This is the project that uses the LSTM model to classify emotions based on input comment. This project is containerized using Docker for easy deployment and execution.

This classification model was trained in comments based on Wikipedia. The expected effect is to classify the most related toxic level for the input article, and there are 6 different level will present after you input your text and calculate the probability between the level and input text, such as toxic, threat, and insult etc.

<p float="left">
  <img src="https://github.com/TTonnyy789/Pictueres/blob/main/LSTM/1200px-Keras_logo.svg.png" width="400" />
  <img src="https://github.com/TTonnyy789/Pictueres/blob/main/Topic_Modelling/docker%20logo1.png" width="400" /> 
</p>

<!-- ![BERTopic Logo](https://github.com/TTonnyy789/Pictueres/blob/main/Topic_Modelling/3632492bb621b51af9c5fccc02da54fe0e44374f-1824x1026.png)![Docker Logo](https://github.com/TTonnyy789/Pictueres/blob/main/Topic_Modelling/docker%20logo1.png) -->

## Table of Contents

- [Data Preprocessing](#data-preprocessing)
- [Model Architecture](#model-architecture)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Training and Validation](#training-and-validation)
- [Results Visualization](#results-visualization)
- [Saving and Loading](#saving-and-loading)
- [Usage](#usage)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Data Preprocessing 
The data is sourced from CSV files. Comments in the dataset undergo a series of preprocessing steps:

- Lowercasing
- Replacing contractions and specific patterns
- Removing non-word characters
- Removing stopwords (using the NLTK library)
- Tokenization (converting text into sequences of integers)

## Model Architecture
The model is constructed using the Keras Sequential API. It comprises:

- Embedding layer: Converts tokenized sequences into dense vectors
- Two LSTM layers: Captures sequential dependencies and patterns in the tokenized sequences
- Dense layer: Outputs probabilities for each class

## Hyperparameter Tuning

The model's performance largely hinges on selecting the right hyperparameters. In this project, we use Random Search to search across a range of values for:

- Embedding layer output dimension
- Number of units in the LSTM layers
- Dropout rates for LSTM layers
- Learning rate for the Adam optimizer

This approach ensures that we don't just rely on intuition or default values but systematically find the best hyperparameters for our dataset.

<img src="https://github.com/TTonnyy789/Pictueres/blob/main/LSTM/hyperparameter_tuning.png" alt="Image1" width="600"/>

## Training and Validation

The dataset is split into training and validation sets. The model is then trained on the training set for a specified number of epochs, while the validation set is used to prevent overfitting and gauge the model's performance on unseen data.

<img src="https://github.com/TTonnyy789/Pictueres/blob/main/LSTM/traing_and_validation.png" alt="Image1" width="600"/>

## Results Visualization

Post-training, we visualize the training and validation loss and AUC (Area Under Curve) across epochs. This helps in understanding how well the model is learning and whether it's overfitting or underfitting.

<img src="https://github.com/TTonnyy789/Pictueres/blob/main/LSTM/results_visulization.png" alt="Image1" width="600"/>

The following two pictures are the result of the AUC and LOSS value on both training data and validation data.

<p float="left">
  <img src="https://github.com/TTonnyy789/Pictueres/blob/main/LSTM/result1.png" width="400" />
  <img src="https://github.com/TTonnyy789/Pictueres/blob/main/LSTM/result2.png" width="400" /> 
</p>

## Saving and Loading

To ensure the model's reusability, both the model and the tokenizer are saved post-training. The saved model can then be loaded and used for predictions without having to undergo training again.

## Usage

### Prerequisites

- Docker installed on your machine.
- Git for cloning this repository.

### Pulling the Docker Image

```bash
docker pull ttonnyy789/lstm-tt:latest
```

### Running on Docker container

```bash
docker run -it --rm ttonnyy789/lstm-tt
```

Once you execute the command, the follow result will present on your terminal. After the `Enter text: (or type 'exit' to quit):` appear you can provide any text input when prompted to classify your comment.


<img src="https://github.com/TTonnyy789/Pictueres/blob/main/LSTM/toxic_example.jpg" alt="Image1" width="600"/>

<!-- ![Docker file run successfully](https://github.com/TTonnyy789/Pictueres/blob/main/Topic_Modelling/input.jpg) -->

## Building the Docker Image Locally (Optional)

If you want to build the Docker image locally especially you are using the M series Apple devices such as M1 Pro Macbook:

The multi-platform docker image builder is required in this case for the purpose of running this docker image successfully on other computer. 
Let's take M1 Apple Silicon device as an example:

```bash
docker buildx create --use
docker buildx inspect --boostrap
```
You can execute the following commands to check this specific docker builder whether has been installed in your machine or not.

```bash
docker images
```

If it is successfully installed, you will be able to find a docker image on your local device called `moby/buildkit`.

Next step, execute the commands below, you would be able to build and run this docker file successfully.

```bash
git clone https://github.com/TTonnyy789/LSTM_wikipedia.git
cd LSTM_wikipedia
```
Once cloned this repositories from Github, you can run following commands and run this dockerized model locally. 

In this case, this image is built with `--platform linux/arm64,linux/amd64` setting, so it would not store image to your local device automatically if you did not add `--load`. 

Therefore, `--load` is essential in this stage, because this docker file is based on multi-platform, the initial configuration of building docker image will not store it directly on you device.

You can change `--load` to `--push` if you want to push this image onto your docker hub.

```bash
docker buildx build --platform linux/amd64,linux/arm64 -t ttonnyy789/bertopic-bb --load .
```
Last but not least, execute this command and enjoy your topic predicting journey ! !

```bash
docker run -it --rm ttonnyy789/bertopic-bb
```

## License
[MIT]((https://choosealicense.com/licenses/mit/))

## Acknowledgements

- Thanks to Docker and the BERTopic community for the foundational tools and resources.
- Special thanks to ChatGPT for project troubleshooting guidance.