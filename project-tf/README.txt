The following are the files in this folder:

prepro_tutorial.py:
preprocesses images

model.py: 
where the model is constructed

train.py: 
To train the model. Need to specify the stage in the constant STAGE, and
the model to use and hyperparameters in FLAGS. The flag best_val_accuracy specifies
the validation accuracy that needs to be exceeded in order to save model parameters. 
The model flag refers to the model, can be "linear" to run linear classifier or "cnn"
to run a random cnn that we made up.
To train the model, call "python train.py" from this folder. The model parameters will 
be saved to the folder stage<STAGE>-weights/<model>/
If there are already weights in stage<STAGE>-weights/<model>/ then running train.py
will automatically load the weights. If you want to start with fresh weights then delete
all the files in stage<STAGE>-weights/<model>/
Basically whenever we change the model we should delete the files from 
stage<STAGE>-weights/<model>/ and also set best_val_accuracy to 0.

test.py
To test on the test set. The constant STAGE and the flags need to all be the exact same
as in train.py for it to work.

stage<STAGE>_solution.csv:
labels for stage <STAGE> 

stage<STAGE>-npy/:
folder containing numpy arrays representing images for stage <STAGE>

stage<STAGE>-weights/:
contains folders linear/ and cnn/ which contain saved model parameters

labels.py: 
random stuff