# task-5

## Steps to Use
1. Clone this repository
2. Download the dataset from [Kaggle](https://www.kaggle.com/competitions/cifar-10/data)
3. In the run.sh file, change the parameters that you want to change as follows
    - --lr initial learning rate for gd based algorithms, float
    - --momentum momentum to be used by momentum based algorithms
    - --num_hidden number of hidden layers
    - --sizes comma separated list for the size of each hidden layer
    - --activation [tanh / sigmoid]
    - --loss [sq / ce]
    - --opt [gd / momentum / nag / adam]
    - --batch_size [1 / multiples of 5]
    - --anneal [true / false]
    - --save_dir path to pickled model
    - --expt_dir path to log files
    - --train path to training data - If the training files are stored in /train, --train /train
    - --test path to test data - If the test files are stored in /test, --test /test

This model has not been tested in its entirety due to the following error generated while loading test data: libpng error: Read Error

Thus it is not a guarantee any code below loading test data will work or not. However, the logs will still be visible on the terminal after executing the bash script.

Nesterov Accerlerated Gradient (nag) is executed properly, however, the logic applied is not guaranteed to be correct, as I could not find any reliable sources for the theory / implementation behind the same.

Reference - [NNFS Book](https://nnfs.io/)