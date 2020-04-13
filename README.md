# covid-2020
ML projects to help in adverse condition of covid-2020

This is the project that uses CNN to train the model to detect if the pnemonia is simple or is affected by corona virus.

This model is not very reliable for public use because it is trained just on 241 images and is tested on 41 images of lungs.

I was struggling with the acccuracy of about 80%, then i changed the test and train images and achieved the higher accuracy og 83%.
I got the accuracy of 83% in 25 epochs

loss: 0.1881 - accuracy: 0.8319 - val_loss: 0.7290 - val_accuracy: 0.7788

This ig great accuracy with only 261 images trained for 25 epochs.

It got higher accuracy on input shape of(64,64) then (512,512)
