# RNN configuration log

## Initial setup

First of all i tried to write a model just like the one that has been shown in the lecture

```Python
    model = tf.keras.models.Sequential(
        [
            # Convolution layer
            tf.keras.layers.Conv2D(
                32, (3, 3), activation="relu", input_shape=(30, 30, 3)
            ),
            # Pooling layer
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

            # Flatten units
            tf.keras.layers.Flatten(),

            # Add a hidden layer with dropout
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.5),

            # Add an output layer with output units for all 44 signs
            tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax"),
        ]
    )
    
    # Train neural network
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
```

And i got **total accuracy of 6% **
There is obiously much room for improvement ;)


## Development epoch #1

First of all I tried to skip dropout layer and check out what will it give me, regardless of oversampling
Also I added another dense layer
``` tf.keras.layers.Dense(128, activation="relu"), ```

As a matter of fact it worked very well, accuracy jumped up to 98% on training data and 94% on testing data
However i suspected overfitting, so I decided to introduce a dropout layers after each dense layer.

## Development epoch #2

Let's play with dropout parameters
Starting from 0.5 and down
```
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.5),
```
**Result: accuracy ~6%. Previous results were definitely an overfit**

Lets try dropout -> 0.25

**Result is mostly the same ~5.3%**

Alright what if i do only one dropout of .2 after the first layer
```
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.20),
            tf.keras.layers.Dense(128, activation="relu"),
```
**Oddly enough that setup results in 89% accuracy**

It seems that dropouts were too aggressive, lets lower them a little more
Current setup:
```
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.10),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.10),
```

**Result is 90.5% on training data and 92% on testing data, seems like it becomes more balanced**

And just for kicks lets add a couple more layers, why not

3 layers with dropout .1
**Accuracy 75% on training data**

The last try did not work too well, yet showed gradual growth each epoch
What if i do 25 epochs? I hope my old cpu can manage

First of all, It would seem that dropout works quite randomly, first epoch gave me 31%, while from the last try I got only 8%

**Result 95% on training data and 93% on testing data**

Okay increasing epochs number works fine, but takes much longer time to train, I do not like that way.
Let's play with other meta parameters

## Development epoch #3

Setting back parameters for now
EPOCHS = 10
Two such layers:

```
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.10),
```

That model is 83% accurate on training data and 87.5% accurate on testing data.

Lets change number of filters in convolution layer 32 ----> 128
**Process: each epoch takes much more time but results in more accuracy. I got 48% in first epoch, yet each step takes 5 times more time to process.**
**Result:  93.5% accuracy. That's better but not worth significant time increase, need to decrease filters.**

Filters 128 ----> 64
Nodes in hidden layers 128--->256
pool_size=(2, 2) -----> (3, 3)
**Result: 95% on training data and 94% on testing data. That's better but can yet be improved.**

## Development epoch #4 --- Let's try different optimizers

So adam yielded 94%
Let's try RMSprop
**Work: okay that is 60% on first epoch**
**Result: 96% training and 97% on testing data**

Let's go with SGD 
**Results: <10%**

RMSprop it is. Let's play with learning rate and momentum

```
    rmsprop_optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9, momentum=0.0, epsilon=1e-07)

    model.compile(
        optimizer=rmsprop_optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
```

    that will be the starting point 
**Accuracy 94.5%**

Learning rate 0.001 ---> 0.005
**Worse**

Momentum 0--->0.05
**97% but much slower**

Momentum 0--->0.01
Rho 0.9 ---> 0.8
Learning rate .001 ---> .002

**Result 90%**

Okay it seems that i found good enough option, that was 97% with this optimizer setup:
```    model.compile(optimizer="RMSprop", loss="categorical_crossentropy", metrics=["accuracy"]) ```

Time to try more advanced methods

## Development epoch #5 --- Advanced methods

We're back to using plain RMSprop:

```
    model.compile(
        optimizer="RMSprop",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
```

Current setup: 
- convolution layer Conv2D(64, (3, 3), activation="relu", input_shape=(30, 30, 3))
- pooling layer with pool size (3, 3)
- 2 dense layers of 256 nodes, relu activation
- dropout 0.1 after each

It yields **accuracy of 94-95%**

Adding early stopping and learning rate scheduling like this:

```
    def step_decay(epoch):
        initial_rate = 0.001
        drop = 0.5
        epochs_drop = 5.0
    return initial_rate * np.power(drop, np.floor((1 + epoch) / epochs_drop))
    early_stopping = EarlyStopping(monitor='accuracy', patience=3)
    lr_schedule = LearningRateScheduler(step_decay)
    model.fit(x_train, y_train, epochs=EPOCHS, callbacks=[early_stopping, lr_schedule])
```

would increase accuracy up to 99%
HOWEVER that is against declared task since it modify main() function

**Result >99% yet unapplicable for this task**

Also i'd like to try different activation functions
First of all - leaky ReLU algorithm

Convolution layer and hidden layers set to leaky ReLU:
**Result 94.4%**

Convolution layer -> swish | hidden layer -> softmax
**Result NO**

Convolution layer -> swish | hidden layer -> leaky ReLU
Yeah softmax for hidden layer doesn't work
**Result 95.4%**

Lets modify metrics:
metrics=["accuracy", "precision", "recall"]
Activations stay the same as last iteration
**Result 96.3% but much slower**

Hidden layers activation ---> swish
**Result 94.2% worse than leaky ReLU**

Alright lets try my PC instead of laptop for some faster learning
Spent around an hour understanding how PowerShell works just to get 10s per epoch speed boost, great.
Lets increase epochs to 50 :)

**Result 98.5%**

Setting learning rate ---> 0.075

**Result 97.6%**

pool_size=(3,3) ---> (2, 2) 
filters 64 ---> 256
epochs ---> 20 

**Result 97.4% and so very long time training** 

Reduce to 128 filters
Remove pooling

**Too long time training**

OK let's turn pooling back on and keep 128 filters

**Still too long**

Reducing # of epochs ----> 10

**Result 96%**

**Results of switching to PC with an actual GPU were not that impressive**
