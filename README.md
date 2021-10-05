# lightweight-motion-forecasting
A lightweight graph-convolutional model for skeletal human motion forecasting on the Human3.6M (H3.6M) dataset.

## Setup

* Install the python libraries ```pip install -r requirements.txt``` (This file contains the GPU libs for tensorflow and tensorflow_graphics, remove '-gpu' to use the cpu versions)

## Usage

* Get the [H3.6M Dataset](http://vision.imar.ro/human3.6m/description.php)
* The CLI is located in ```main.py```, it consists of two subprograms ```train``` and ```eval``` for training and evaluation of models respectively.
  * To train a model, call ```python main.py train```
    * This will train a model with the default configuration (s. ```configs.py```)
  * To evaluate a model, call ```python main.py eval --checkpoint <path_to_checkpoint>```
    * This will run the default evaluation on a model with the default configuration (s. ```configs.py```), restored from the checkpoint thats passed in ```path_to_checkpoint```
  * Alternatively you can alter the defaults by passing additional cli arguments
