import uuid
import tensorflow as tf
import numpy as np

# Model class
class SpeedLimitCNN:

  # Initialize model
  def __init__(self, train_data, test_data, valid_data):
    # Inputs and labels
    self.train_inputs = [sample["image"] for sample in train_data]
    self.train_labels = [int(sample["label"]) for sample in train_data]
    self.test_inputs = [sample["image"] for sample in test_data]
    self.test_labels = [int(sample["label"]) for sample in test_data]
    self.valid_inputs = [sample["image"] for sample in valid_data]
    self.valid_labels = [int(sample["label"]) for sample in valid_data]
    self.model_dir = dir + 'models/'

    print(f"Model dir: {self.model_dir}")

    self.image_dims = np.array(train_data[0]['image']).shape

    labels = np.unique(self.train_labels)

    """
    One hot encode labels.

    [1,0,0,0,0,0] = 25mph
    [0,1,0,0,0,0] = 30mph
    [0,0,1,0,0,0] = 35mph
    [0,0,0,1,0,0] = 40mph
    [0,0,0,0,1,0] = 45mph
    [0,0,0,0,0,1] = 50mph
    """
    _train_labels = []
    for j in range(len(self.train_labels)):
      for i, label in enumerate(labels):
        if self.train_labels[j] == label:
          one_hot = np.zeros(len(labels), dtype=int)
          one_hot[i] = 1
          _train_labels.append(one_hot)
    self.train_labels = _train_labels

    _test_labels = []
    for j in range(len(self.test_labels)):
      for i, label in enumerate(labels):
        if self.test_labels[j] == label:
          one_hot = np.zeros(len(labels), dtype=int)
          one_hot[i] = 1
          _test_labels.append(one_hot)
    self.test_labels = _test_labels

    _valid_labels = []
    for j in range(len(self.valid_labels)):
      for i, label in enumerate(labels):
        if self.valid_labels[j] == label:
          one_hot = np.zeros(len(labels), dtype=int)
          one_hot[i] = 1
          _valid_labels.append(one_hot)
    self.valid_labels = _valid_labels

    # Hyperparameters
    self.epochs = 750                       #   Number of training epochs
    self.batch_size = 10                    #   Number of samples to train per epoch
    self.n_hidden = 4096                    #   Number of nodes in the first hidden layer
    self.n_outputs = 6                      #   6 outputs, 1 for each speed limit
    self.hidden_activation = 'relu'         #   Activation function for hidden layers
    self.output_activation = 'softmax'      #   Output activation (softmax because we want a multi-class probability distribution as our output)
    self.learning_rate = 0.000105           #   Learning rate for training the model
    self.m1_decay = 0.99                    #   First moments decay (specifically for Adam and Adamax optimizers)
    self.m2_decay = 0.999                   #   Second moments decay (also specifically for Adam/Adamax optimizers)
    self.stabilizer = 1e-07                 #   Numerical constant for avoiding 0s and stabilizing parameter updates
    self.n_filters = 32                     #   Number of convolutional filters to apply/train
    self.filter_shape = [3,3]               #   Shape of kernel for conv. filter
    self.max_pool_shape = [4,4]             #   Shape of matrix for applying max pooling 
    self.avg_pool_shape = [3,3]             #   Shape of matrix for applying average pooling

    # Configure training algorithim (optimizer)
    self.optimizer = tf.keras.optimizers.Adam(
                                                self.learning_rate,
                                                self.m1_decay,
                                                self.m2_decay,
                                                self.stabilizer
                                                )
    
    """
      Set loss function. 
      
      Catergorical Cross-Entropy was chosen because it provides a relative ratio of 
      misclassified labels in a multi-class classification task. Basically, it's
      a metric that represents how many samples of each class are labelled correctly/incorrectly
      relative to the entire training set. 

      Each class can be thought of as a 'category'
    """
    self.loss_func = tf.keras.losses.CategoricalCrossentropy()


    # Model
    self.model = tf.keras.models.Sequential([
        # Apply first set of convolutional filters to the inputs
        tf.keras.layers.Conv2D(self.n_filters,self.filter_shape,activation=self.hidden_activation,input_shape=(self.image_dims[0],self.image_dims[1],1),padding='same'),

        # Apply max pooling to extract sharper parts of the images such as edges
        tf.keras.layers.MaxPool2D(self.max_pool_shape),

        # Apply another set of conv. filters to learn from the features extracted from the max pool layer
        tf.keras.layers.Conv2D(self.n_filters,self.filter_shape,activation=self.hidden_activation,input_shape=(self.image_dims[0],self.image_dims[1],1),padding='same'),

        # Apply average pooling to extract smoother parts of the image such as contours and rounded shapes
        tf.keras.layers.AveragePooling2D(self.avg_pool_shape),

        # Upsample resulting image to help preserve extracted features
        tf.keras.layers.UpSampling2D(size=1),

        # Flatten inputs for feeding into the DNN
        tf.keras.layers.Flatten(),

        # First DNN hidden layer
        tf.keras.layers.Dense(units = self.n_hidden, activation = self.hidden_activation, kernel_regularizer='l2'),

        # Second hidden layer
        tf.keras.layers.Dense(units = self.n_hidden/2, activation = self.hidden_activation, kernel_regularizer='l2'),

        # Third hidden layer
        tf.keras.layers.Dense(units = self.n_hidden/4, activation = self.hidden_activation, kernel_regularizer='l2'),

        # Third hidden layer
        tf.keras.layers.Dense(units = self.n_hidden/8, activation = self.hidden_activation, kernel_regularizer='l2'),

        # Output layer
        tf.keras.layers.Dense(units = self.n_outputs, activation = self.output_activation)
    ])

    # Generate ID for this model
    self.model_id = uuid.uuid4()


  # Reshape data for use as inputs into the model
  def reshape_data(self, data, image_shape=64):
    return np.reshape(data, [-1, image_shape, image_shape])

  # Train the model
  def train(self):
    # Compile model with optimizer,loss function, and accuracy metrics
    self.model.compile(optimizer=self.optimizer, loss=self.loss_func, metrics=['categorical_accuracy'])
    
    print(f"\n\nTRAINING MODEL...\n")

    # Cast data to numpy array's so tensorflow stops yelling at me
    self.train_inputs = np.array(self.train_inputs)
    self.train_labels = np.array(self.train_labels)
    self.valid_inputs = np.array(self.valid_inputs)
    self.valid_labels = np.array(self.valid_labels)
    self.test_inputs = np.array(self.test_inputs)
    self.test_labels = np.array(self.test_labels)

    # Train model
    self.model.fit(
        x = self.train_inputs,                    # Training inputs
        y = self.train_labels,                    # Training labels,
        epochs = self.epochs,                     # Set number of training epochs
        verbose = 2,                              # How much information to print out about training progress
        shuffle = True,                           # Shuffle training data between epochs
        batch_size = self.batch_size,             # Set number of samples to train per epoch
        validation_data = [self.reshape_data(self.valid_inputs), self.valid_labels],
        use_multiprocessing = True
      )

    print(f"\nEVALUATING MODEL...")
    # Evaluate model on test set
    loss, acc = self.model.evaluate(self.reshape_data(self.test_inputs), self.test_labels)
    threshold = 0.9

    # Save model
    if acc > threshold:
      model_name = f"SpeedLimitCNN_Acc-{round(acc*100.0)}pct.model"
      tf.keras.models.save_model(self.model, self.model_dir + model_name)
      print(f"\nMODEL SAVED TO {self.model_dir+model_name}")

      print(f"\n{self.model.summary()}")
    else:
      print(f"\nMODEL FAILED TO ACHIEVE {round(threshold*100.0,2)}% ACCURACY OR HIGHER")


  # Load model from file
  def load_model(self, filename):
    self.model = tf.keras.models.load_model(filename)

    print(f"\n\nMODEL LOADED!\n{self.model.summary()}")

  
  def predict(self, inputs):
    # Cast data to numpy array's so tensorflow stops yelling at me
    inputs = np.array(inputs)
    predictions = []
    labels = [25,30,35,40,45,50]

    # Predict
    outputs = self.model.predict(self.reshape_data(inputs))

    # Iterate through outputs
    for output in outputs:
      # Round outputs
      output = np.round_(output)

      # Convert one hot encoded output to an integer prediction and add to array
      label_idx = np.argmax(output)
      predictions.append(labels[label_idx])

    # Return predictions
    return predictions