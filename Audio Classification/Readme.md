## Audio Classification
Audio Feature extraction and deep learning-based audio classifier using Different Sequence Models.<br>
This project focuses on designing and implementing a deep learning-based approach to classify audio clips into predefined categories based on the type of sound they contain. <br>
#### Feature Engineering
As data was fairly balanced across all classes having 14 instances of each class in training data and 3 instances in the validation class, no data imbalance was found.
The next step was to extract different features from given .wav data files using a Spectrogram of audio. From the spectrogram, Mel-frequency cepstral coefficients (MFCC), Spectral Centroid, Zero Crossing Rate, Chroma Frequencies, and Spectral Roll-off were calculated using the Librosa package.
In another attempt only MFCC of audio using different n_fft and hope_length as input features to sequence models.
Finally, a pre-trained YAMNet model was used to extract the embedding of audio and used as an input feature for sequence models.
#### Model Architecture
1. Simple RNN:
Initial models incorporated simple RNN architectures with dual layers and experimented with varying learning rates, regularization techniques, and epochs. Validation accuracy plateaued between 40-50%, with no marked improvement upon further experimentation.
2. LSTM Models:
All the different categories of input features were trained using a combination of LSTM and the Dense layer model. Different configurations of layers, the number of units in each layer, different activation functions, and learning rates were experimented with for epochs of 100,200 and sometimes even 500.
The best results were given by 2 LSTM layers and 2 Dense layer networks with “tanh” activation for the LSTM layer “real” for the dense layer and “softmax” for the final layer with 20 to 50 epochs of training. An f-1 score of 0.73 was obtained on validation data and 0.76 on test data using these network architectures.
3. Bidirectional LSTM Models:
Using Bidirectional LSTM instead of the LSTM layer significantly increased performance even with less training. Training a 2-layer bidirectional LSTM network on YAMNet embeddings for 20-30 epochs inferred an F-1 score of 0.80 on validation data and 0.84 on test data.
##### Hyper Parameter Tuning
* Models: Among the SImpleRNN, LSTM, and Bidirectional LSTM, the Bidirectional LSTM performed the best achieving a 0.8437 F-1 score on CLP for 50% of the test data.
* Sampling Rate for embedding: Tuning sampling rate of audio file during feature extraction to 16000, 22000, 44100. Here no significant performance differences were noticed
* Batch size: 32, 64
* Activation functions: relu, tanh
* Patients: 5,10, 20, None
* Regularization: L2 regularization and Dropout
  
Reference:<br>
https://www.kdnuggets.com/2020/02/audio-data-analysis-deep-learning-python-part-1.html <br>
https://www.tensorflow.org/tutorials/audio/transfer_learning_audio