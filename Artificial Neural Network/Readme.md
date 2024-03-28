# Artificial Neural Network with Python
This project focuses on the development and training of feed-forward neural networks to predict whether a given antibacterial peptide also possesses antibiofilm properties.
This project explores the effectiveness of various neural network configurations, activation functions, and regularization techniques in handling imbalanced datasets.<br>
#### Approach and Methodology
1. <b>Feature Engineering</b><br>
In the process of feature engineering, two primary models were employed: a Bag of Words model and a k-mer model with k set to 2. Working with k-mer with k = 3, which led to an over sparse matrix and very high training time even with tensorflow`s sparse matrix representation with no major improvement on validation data.
Additionally, to address class imbalance, oversampling of the minority class (class 1) was implemented for both models, aiming to enhance the model's ability to generalize over the minority instances effectively. Oversampling the minority class to 50% of the major class and 100% of the major class was implemented.
2. <b>Neural Network Architecture</b><br>
The neural network's initial configuration comprised a first layer with 8 neurons, a second layer with 4 neurons, and a final output layer with a single neuron, tailored for binary classification tasks. Despite experimenting with more complex architectures by increasing both the number of layers and neurons per layer (eg. [216, 128, 32, 8, 1] for k-mer model ), such adjustments did not yield improvements in model performance. This observation suggests that the simpler architecture was more suited to our dataset, as complexity led to fitting issues and did not enhance the Matthews Correlation Coefficient (MCC).
3. <b>Loss Function</b><br>
Binary cross-entropy was selected as the loss function, appropriate for the binary classification problem at hand.
4. <b>Optimization Techniques</b><br>
Regarding optimization techniques, both gradient descent and the Adam optimizer were evaluated. Experiments with RMSProp, L1 and L2 regularization were also conducted.The Adam optimizer with L2 regularization for the Python model demonstrated superior performance, especially when applied to the dataset processed with a k-mer value of 2, overcoming the vanishing gradients issue observed with gradient descent.
5. <b>Training Iterations</b><br>
Python models were subjected to training for between 1000 and 5000 iterations. Extending training beyond this range did not result in noticeable improvements in the MCC for validation data or the CLP benchmark.
For TensorFlow models, the training was capped at 500 to 1000 epochs based on observations that, post approximately 1000 epochs, minimal to no significant gains in test MCC scores were noted. However, certain instances exhibited slight improvements, justifying the decision to set the epoch count to 500 for TensorFlow models.
6. <b>Data Splitting</b><br>
The base dataset, comprising 1566 samples (1424 of class -1 and 142 of class 1), was divided, reserving 20% for validation purposes. This split was maintained even after oversampling the minority class, resulting in an equal number of class -1 and class 1 samples (1424 each), from which 20% was allocated for validation to ensure a balanced representation of classes in the training and validation sets.