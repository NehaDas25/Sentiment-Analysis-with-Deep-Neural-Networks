# Work Report

## Information

- Name: <ins> DAS, NEHA </ins>
- GitHub: <ins> NehaDas25 </ins>


## Features

- Not Implemented:
  - what features have been implemented

<br><br>

- Implemented:
  - PART 2: Importing the Data
  - PART 2.3: Converting a Tweet to a Tensor
    - Exercise 1: tweet_to_tensor
      - Implemented a function called tweet_to_tensor() that takes tweet, vocab_dict, unk_token='__UNK__', verbose=False as inputs.
      - Returns an array of numbers which is stored in tensor_l(a python list).
      - Maped each word in tweet to corresponding token in 'Vocab'.
      - Used Python's Dictionary.get(key,value) so that the function returns a default value if the key is not found in the dictionary.
      - This passed 4 test cases and failed 4 test cases.
    
  - PART 2.4: Creating a Batch Generator
    - Exercise 2: data_generator
       - Implemented a function data_generator() that takes data_pos, data_neg, batch_size, loop, vocab_dict, shuffle=False as inputs.
       - This generator function returns the data in a format (tensors) that you could directly use in your model.
       - It returns a triplet: the inputs, targets, and loss weights:
        1. Inputs: is a tensor that contains the batch of tweets we put into the model.
        2. Targets: is the corresponding batch of labels that we train to generate.
        3. Loss weights: here are just 1s with same shape as targets.
       - This data generator can be used to create a data generator for the training data, and another data generator for the validation data and a third data generator that does not loop, for testing the final accuracy of the model.
       - This passed 13 test cases and failed 1 test case.

  - PART 3: Defining Classes
  - PART 3.1: ReLU Class
    - Exercise 3: Relu
      - Using base class which is called **Layer class**, we implemented a **Relu class**.
      - In Relu Class a ReLU activation function forward() was implemented that takes self, x as inputs.
      - Our function take in a matrix or vector and transformed all the negative numbers into 0 while keeping all the positive numbers intact.
      - Used numpy.maximum(A,k) to find the maximum between each element in A and a scalar k 
      - This passed all the test cases.  
  
  - PART 3.2: Dense Class
    - Exercise 4: Dense
      - Using base class which is called **Layer class**, we implemented a **Dense class**.
      - In Dense Class a forward() function that takes self, x as inputs and init_weights_and_state() that takes self, input_signature, random_key as inputs were implemented.
      - The forward function multiplies the input to the layer (x) by the weight matrix (W).Used numpy.dot to perform the matrix multiplication.
      - Implemented the weight initializer new_weights function
       1. Weights are initialized with a random key.
       2. The second parameter is a tuple for the desired shape of the weights (num_rows, num_cols)
       3. The num of rows for weights should equal the number of columns in x, because for forward propagation, you will multiply x times weights.
      - This passed all the test cases as well.

  - PART 3.3: Model - Before implementing the classifier function, for the model implementation, we will use the Trax layers module, imported as tl. The layers needed are Dense, Serial, Embedding, Mean, LogSoftmax.
    - Exercise 5: classifier
      - Implemented a function classifier() that takes vocab_size=9088, embedding_dim=256, output_dim=2, mode='train' as inputs.
      - Created an Embedding layer using tl.Embedding(vocab_size=vocab_size,d_feature=embedding_dim).
      - Created a mean layer, to create an "average" word embedding using tl.Mean(axis=1).
      - Created the log softmax layer (no parameters needed), using tl.LogSoftmax().
      - Use tl.Serial to combine all layers and create the classifier of type trax.layers.combinators.Serial, model = tl.Serial(embed_layer,mean_layer,dense_output_layer,log_softmax_layer).
      - This passed all the test cases as well.

  - PART 4: Training - TrainTask, EvalTask and Loop has been defined in preparation to train the model.
  - PART 4.1: Training the Model
    - Exercise 6: train_model
      - Implemented a function train_model() that takes classifier, train_task, eval_task, n_steps, output_dir as inputs.
      - To train the model (classifier that was implemented earlier) for the given number of training steps (n_steps) using TrainTask, EvalTask and Loop.
      - For the EvalTask, the eval_task is passed as a list explicitly.
      - This passed all the test cases as well.
  
  - PART 5: Evaluation
  - PART 5.1: Computing the Accuracy on a Batch
    - Exercise 7: compute_accuracy
      - Implemented a function compute_accuracy() that takes preds, y, y_weights as inputs.
      - This function evaluates our model on the validation set and returns the accuracy, weighted_num_correct, sum_weights.
      - preds contains the predictions.
       1. Its dimensions are (batch_size, output_dim). output_dim is two in this case. Column 0 contains the probability that the tweet belongs to class 0 (negative sentiment). Column 1 contains probability that it belongs to class 1 (positive sentiment).
       2. If the probability in column 1 is greater than the probability in column 0, then interpret this as the model's prediction that the example has label 1 (positive sentiment).
       3. Otherwise, if the probabilities are equal or the probability in column 0 is higher, the model's prediction is 0 (negative sentiment).
      - y contains the actual labels.
      - y_weights contains the weights to give to predictions.
      - Convert weighted_num_correct and sum_weights as np.float32.
      - This passed all the test cases as well.
 
  - PART 5.2: Testing your Model on Validation Data
    - Exercise 8: test_model
      - Implemented a function test_model() that takes generator, model, compute_accuracy() as inputs.
      - Computed the accuracy over all the batches in the validation iterator.
      - Used compute_accuracy(), which was recently implemented, and returns the overall accuracy.
      - batch has 3 elements:
       1. the first element contains the inputs
       2. the second element contains the targets
       3. the third element contains the weights
      - This passed all the test cases as well.



<br><br>

- Partly implemented:
  - utils.py which contains process_tweet(tweet), load_tweets() and class Layer(object) has not been implemented, it was provided.
  - w1_unittest.py was also not implemented as part of assignment to pass all unit-tests for the graded functions().
  - testing_support_files, model, model_test were also not implemented as part of assignment to pass all unit-tests for the graded functions().

<br><br>

- Bugs
  - In exercise 2.1, 4 test cases passed and 4 test cases failed. 
  - In exercise 2.2, 13 test cases passed and 1 test cases failed.
  - Between the expected output and actual output there is a difference of 1 which is mostly because of extra words in vocab_dict.

<br><br>


## Reflections

- Assignment is very good. Gives a thorough understanding of the basis of tweet_to_tensor, Defining Classes, Neural Networks, train_task, eval_task.


## Output

### output:

<pre>
<br/><br/>
Out[4] -

DeviceArray(5., dtype=float32, weak_type=True)
<class 'jaxlib.xla_extension.DeviceArray'>
Notice that trax.fastmath.numpy returns a DeviceArray from the jax library.

Out[6] - 

f(a) for a=5.0 is 25.0

Out[7] - function

Out[8] - DeviceArray(10., dtype=float32, weak_type=True)

Out[10] - 

The number of positive tweets: 5000
The number of negative tweets: 5000
length of train_x 8000
length of val_x 2000

Out[11] - 

original tweet at training position 0
#FollowFriday @France_Inte @PKuchly57 @Milipol_Paris for being top engaged members in my community this week :)
Tweet at training position 0 after processing:
['followfriday', 'top', 'engag', 'member', 'commun', 'week', ':)']

Out[26] -

Actual tweet is
 Bro:U wan cut hair anot,ur hair long Liao bo
Me:since ord liao,take it easy lor treat as save $ leave it longer :)
Bro:LOL Sibei xialan

Tensor of tweet:
 [1064, 136, 478, 2351, 744, 8149, 1122, 744, 53, 2, 2671, 790, 2, 2, 348, 600, 2, 3488, 1016, 596, 4558, 9, 1064, 157, 2, 2]

Expected output
Actual tweet is
 Bro:U wan cut hair anot,ur hair long Liao bo
Me:since ord liao,take it easy lor treat as save $ leave it longer :)
Bro:LOL Sibei xialan

Tensor of tweet:
 [1065, 136, 479, 2351, 745, 8148, 1123, 745, 53, 2, 2672, 791, 2, 2, 349, 601, 2, 3489, 1017, 597, 4559, 9, 1065, 157, 2, 2]

Out[24] - All tests passed 

Out[29] - 

Inputs: [[2005 4450 3200    9    0    0    0    0    0    0    0]
 [4953  566 2000 1453 5173 3498  141 3498  130  458    9]
 [3760  109  136  582 2929 3968    0    0    0    0    0]
 [ 249 3760    0    0    0    0    0    0    0    0    0]]
Targets: [1 1 0 0]
Example Weights: [1 1 1 1]

Out[30] - 

The inputs shape is (4, 14)
input tensor: [3 4 5 6 7 8 9 0 0 0 0 0 0 0]; target 1; example weights 1
input tensor: [10 11 12 13 14 15 16 17 18 19 20  9 21 22]; target 1; example weights 1
input tensor: [5737 2900 3760    0    0    0    0    0    0    0    0    0    0    0]; target 0; example weights 1
input tensor: [ 857  255 3651 5738  306 4457  566 1229 2766  327 1201 3760    0    0]; target 0; example weights 1
Expected output
The inputs shape is (4, 14)
input tensor: [3 4 5 6 7 8 9 0 0 0 0 0 0 0]; target 1; example weights 1
input tensor: [10 11 12 13 14 15 16 17 18 19 20  9 21 22]; target 1; example weights 1
input tensor: [5738 2901 3761    0    0    0    0    0    0    0    0    0    0    0]; target 0; example weights 1
input tensor: [ 858  256 3652 5739  307 4458  567 1230 2767  328 1202 3761    0    0]; target 0; example weights 1

Out[31] - All tests passed 

Out[33] - 

Test data is:
[[-2. -1.  0.]
 [ 0.  1.  2.]]
Output of Relu is:
[[0. 0. 0.]
 [0. 1. 2.]]
Expected Outout
Test data is:
[[-2. -1.  0.]
 [ 0.  1.  2.]]
Output of Relu is:
[[0. 0. 0.]
 [0. 1. 2.]]

 Out[34] - 

 All tests passed

 Out[36] - 

The random seed generated by random.get_prng
DeviceArray([0, 1], dtype=uint32)
choose a matrix with 2 rows and 3 columns
(2, 3)
Weight matrix generated with a normal distribution with mean 0 and stdev of 1
DeviceArray([[ 0.95730704, -0.9699289 ,  1.0070665 ],
             [ 0.3661903 ,  0.1729483 ,  0.29092234]], dtype=float32)

Out[40] - 

Weights are
  [[-0.02837107  0.09368163 -0.10050073  0.14165013  0.10543301  0.09108127
  -0.04265671  0.0986188  -0.05575324  0.0015325 ]
 [-0.2078568   0.05548371  0.09142365  0.05744596  0.07227863  0.01210618
  -0.03237354  0.16234998  0.02450039 -0.13809781]
 [-0.06111237  0.01403725  0.08410043 -0.10943579 -0.1077502  -0.11396457
  -0.0593338  -0.01557651 -0.03832145 -0.11144515]]
Foward function output is  [[-3.0395489   0.9266805   2.5414748  -2.0504727  -1.9769386  -2.5822086
  -1.7952733   0.94427466 -0.89803994 -3.7497485 ]]
Expected Outout
Weights are
  [[-0.02837108  0.09368162 -0.10050076  0.14165013  0.10543301  0.09108126
  -0.04265672  0.0986188  -0.05575325  0.00153249]
 [-0.20785688  0.0554837   0.09142365  0.05744595  0.07227863  0.01210617
  -0.03237354  0.16234995  0.02450038 -0.13809784]
 [-0.06111237  0.01403724  0.08410042 -0.1094358  -0.10775021 -0.11396459
  -0.05933381 -0.01557652 -0.03832145 -0.11144515]]
Foward function output is  [[-3.0395496   0.9266802   2.5414743  -2.050473   -1.9769388  -2.582209
  -1.7952735   0.94427425 -0.8980402  -3.7497487 ]]

Out[42] - 

All tests passed

Out[46] - 

Embedding_3_2

Out[47] - 

Shape of returned array is (2, 3, 2)
DeviceArray([[[-0.09254155,  1.1765094 ],
              [ 1.0511576 ,  0.7154667 ],
              [ 0.7439485 , -0.81590366]],

             [[ 0.7439485 , -0.81590366],
              [ 0.7439485 , -0.81590366],
              [-0.09254155,  1.1765094 ]]], dtype=float32)

Out[53] - 

<class 'trax.layers.combinators.Serial'>
Serial[
  Embedding_9089_256
  Mean
  Dense_2
  LogSoftmax
]
Expected Outout
<class 'trax.layers.combinators.Serial'>
Serial[
  Embedding_9088_256
  Mean
  Dense_2
  LogSoftmax
]

Out[54] - 

All tests passed

Out[60] - 

Serial[
  Embedding_9088_256
  Mean
  Dense_2
  LogSoftmax
]

Out[63] - 

Step      1: Total number of trainable weights: 2327042
Step      1: Ran 1 train steps in 2.08 secs
Step      1: train WeightedCategoryCrossEntropy |  0.68989831
Step      1: eval  WeightedCategoryCrossEntropy |  0.69806957
Step      1: eval      WeightedCategoryAccuracy |  0.43750000

Step     10: Ran 9 train steps in 6.59 secs
Step     10: train WeightedCategoryCrossEntropy |  0.64247584
Step     10: eval  WeightedCategoryCrossEntropy |  0.53418100
Step     10: eval      WeightedCategoryAccuracy |  0.93750000

Step     20: Ran 10 train steps in 1.89 secs
Step     20: train WeightedCategoryCrossEntropy |  0.45600957
Step     20: eval  WeightedCategoryCrossEntropy |  0.33223987
Step     20: eval      WeightedCategoryAccuracy |  1.00000000

Step     30: Ran 10 train steps in 1.42 secs
Step     30: train WeightedCategoryCrossEntropy |  0.24014242
Step     30: eval  WeightedCategoryCrossEntropy |  0.15884450
Step     30: eval      WeightedCategoryAccuracy |  1.00000000

Step     40: Ran 10 train steps in 0.79 secs
Step     40: train WeightedCategoryCrossEntropy |  0.13276729
Step     40: eval  WeightedCategoryCrossEntropy |  0.06164266
Step     40: eval      WeightedCategoryAccuracy |  1.00000000

Step     50: Ran 10 train steps in 1.40 secs
Step     50: train WeightedCategoryCrossEntropy |  0.08444289
Step     50: eval  WeightedCategoryCrossEntropy |  0.06003656
Step     50: eval      WeightedCategoryAccuracy |  1.00000000

Step     60: Ran 10 train steps in 0.76 secs
Step     60: train WeightedCategoryCrossEntropy |  0.04531727
Step     60: eval  WeightedCategoryCrossEntropy |  0.02509754
Step     60: eval      WeightedCategoryAccuracy |  1.00000000

Step     70: Ran 10 train steps in 0.78 secs
Step     70: train WeightedCategoryCrossEntropy |  0.03989115
Step     70: eval  WeightedCategoryCrossEntropy |  0.00249659
Step     70: eval      WeightedCategoryAccuracy |  1.00000000

Step     80: Ran 10 train steps in 0.80 secs
Step     80: train WeightedCategoryCrossEntropy |  0.01885000
Step     80: eval  WeightedCategoryCrossEntropy |  0.00504306
Step     80: eval      WeightedCategoryAccuracy |  1.00000000

Step     90: Ran 10 train steps in 0.93 secs
Step     90: train WeightedCategoryCrossEntropy |  0.04065781
Step     90: eval  WeightedCategoryCrossEntropy |  0.00822989
Step     90: eval      WeightedCategoryAccuracy |  1.00000000

Step    100: Ran 10 train steps in 1.52 secs
Step    100: train WeightedCategoryCrossEntropy |  0.01506269
Step    100: eval  WeightedCategoryCrossEntropy |  0.09649469
Step    100: eval      WeightedCategoryAccuracy |  0.93750000
Expected output (Approximately)
Step      1: Total number of trainable weights: 2327042
Step      1: Ran 1 train steps in 1.79 secs
Step      1: train WeightedCategoryCrossEntropy |  0.69664621
Step      1: eval  WeightedCategoryCrossEntropy |  0.70276678
Step      1: eval      WeightedCategoryAccuracy |  0.43750000

Step     10: Ran 9 train steps in 9.90 secs
Step     10: train WeightedCategoryCrossEntropy |  0.65194851
Step     10: eval  WeightedCategoryCrossEntropy |  0.55310017
Step     10: eval      WeightedCategoryAccuracy |  0.87500000

Step     20: Ran 10 train steps in 3.03 secs
Step     20: train WeightedCategoryCrossEntropy |  0.47625321
Step     20: eval  WeightedCategoryCrossEntropy |  0.35441157
Step     20: eval      WeightedCategoryAccuracy |  1.00000000

Step     30: Ran 10 train steps in 1.97 secs
Step     30: train WeightedCategoryCrossEntropy |  0.26038250
Step     30: eval  WeightedCategoryCrossEntropy |  0.17245120
Step     30: eval      WeightedCategoryAccuracy |  1.00000000

Step     40: Ran 10 train steps in 0.92 secs
Step     40: train WeightedCategoryCrossEntropy |  0.13840821
Step     40: eval  WeightedCategoryCrossEntropy |  0.06517925
Step     40: eval      WeightedCategoryAccuracy |  1.00000000

Step     50: Ran 10 train steps in 1.87 secs
Step     50: train WeightedCategoryCrossEntropy |  0.08931129
Step     50: eval  WeightedCategoryCrossEntropy |  0.05949062
Step     50: eval      WeightedCategoryAccuracy |  1.00000000

Step     60: Ran 10 train steps in 0.95 secs
Step     60: train WeightedCategoryCrossEntropy |  0.04529145
Step     60: eval  WeightedCategoryCrossEntropy |  0.02183468
Step     60: eval      WeightedCategoryAccuracy |  1.00000000

Step     70: Ran 10 train steps in 0.95 secs
Step     70: train WeightedCategoryCrossEntropy |  0.04261621
Step     70: eval  WeightedCategoryCrossEntropy |  0.00225742
Step     70: eval      WeightedCategoryAccuracy |  1.00000000

Step     80: Ran 10 train steps in 0.97 secs
Step     80: train WeightedCategoryCrossEntropy |  0.02085698
Step     80: eval  WeightedCategoryCrossEntropy |  0.00488479
Step     80: eval      WeightedCategoryAccuracy |  1.00000000

Step     90: Ran 10 train steps in 1.00 secs
Step     90: train WeightedCategoryCrossEntropy |  0.04042089
Step     90: eval  WeightedCategoryCrossEntropy |  0.00711416
Step     90: eval      WeightedCategoryAccuracy |  1.00000000

Step    100: Ran 10 train steps in 1.79 secs
Step    100: train WeightedCategoryCrossEntropy |  0.01717071
Step    100: eval  WeightedCategoryCrossEntropy |  0.10006869
Step    100: eval      WeightedCategoryAccuracy |  0.93750000

Out[64] - 

Step      1: Total number of trainable weights: 2327042
Step      1: Ran 1 train steps in 1.70 secs
Step      1: train WeightedCategoryCrossEntropy |  0.68741310
Step      1: eval  WeightedCategoryCrossEntropy |  0.67939472
Step      1: eval      WeightedCategoryAccuracy |  0.56250000

Step     10: Ran 9 train steps in 5.41 secs
Step     10: train WeightedCategoryCrossEntropy |  0.63954264
Step     10: eval  WeightedCategoryCrossEntropy |  0.54660094
Step     10: eval      WeightedCategoryAccuracy |  1.00000000
 All tests passed

Out[65] - 

The batch is a tuple of length 3 because position 0 contains the tweets, and position 1 contains the targets.
The shape of the tweet tensors is (16, 15) (num of examples, length of tweet tensors)
The shape of the labels is (16,), which is the batch size.
The shape of the example_weights is (16,), which is the same as inputs/targets size.

Out[66] - 

The prediction shape is (16, 2), num of tensor_tweets as rows
Column 0 is the probability of a negative sentiment (class 0)
Column 1 is the probability of a positive sentiment (class 1)

View the prediction array
DeviceArray([[-9.5181179e+00, -7.3432922e-05],
             [-7.9278040e+00, -3.6072731e-04],
             [-1.0872704e+01, -1.9073486e-05],
             [-7.3449020e+00, -6.4611435e-04],
             [-5.5219045e+00, -4.0061474e-03],
             [-8.2074995e+00, -2.7275085e-04],
             [-9.1085110e+00, -1.1062622e-04],
             [-7.3503180e+00, -6.4253807e-04],
             [-2.3729801e-03, -6.0448222e+00],
             [-3.0136108e-04, -8.1069403e+00],
             [-1.1029243e-03, -6.8103848e+00],
             [-1.9073486e-06, -1.3039824e+01],
             [-2.3144841e-02, -3.7775359e+00],
             [-5.7449341e-03, -5.1623096e+00],
             [-2.2547245e-03, -6.0958142e+00],
             [-1.5783310e-04, -8.7545252e+00]], dtype=float32)

Out[67] - 

Neg log prob -9.5181	Pos log prob -0.0001	 is positive? True	 actual 1
Neg log prob -7.9278	Pos log prob -0.0004	 is positive? True	 actual 1
Neg log prob -10.8727	Pos log prob -0.0000	 is positive? True	 actual 1
Neg log prob -7.3449	Pos log prob -0.0006	 is positive? True	 actual 1
Neg log prob -5.5219	Pos log prob -0.0040	 is positive? True	 actual 1
Neg log prob -8.2075	Pos log prob -0.0003	 is positive? True	 actual 1
Neg log prob -9.1085	Pos log prob -0.0001	 is positive? True	 actual 1
Neg log prob -7.3503	Pos log prob -0.0006	 is positive? True	 actual 1
Neg log prob -0.0024	Pos log prob -6.0448	 is positive? False	 actual 0
Neg log prob -0.0003	Pos log prob -8.1069	 is positive? False	 actual 0
Neg log prob -0.0011	Pos log prob -6.8104	 is positive? False	 actual 0
Neg log prob -0.0000	Pos log prob -13.0398	 is positive? False	 actual 0
Neg log prob -0.0231	Pos log prob -3.7775	 is positive? False	 actual 0
Neg log prob -0.0057	Pos log prob -5.1623	 is positive? False	 actual 0
Neg log prob -0.0023	Pos log prob -6.0958	 is positive? False	 actual 0
Neg log prob -0.0002	Pos log prob -8.7545	 is positive? False	 actual 0

Out[68] - 

Array of booleans
DeviceArray([ True,  True,  True,  True,  True,  True,  True,  True,
             False, False, False, False, False, False, False, False],            dtype=bool)
Array of integers
DeviceArray([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=int32)
Array of floats
DeviceArray([1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0.,
             0.], dtype=float32)

Out[69] - 

True == 1: True
True == 2: False
False == 0: True
False == 2: False

Out[75] - 

Model's prediction accuracy on a single training batch is: 100.0%
Weighted number of correct predictions 64.0; weighted number of total observations predicted 64
Expected output (Approximately)
Model's prediction accuracy on a single training batch is: 100.0%
Weighted number of correct predictions 64.0; weighted number of total observations predicted 64

Out[76] - 

All tests passed

Out[78] - 

The accuracy of your model on the validation set is 0.9960
Expected Output (Approximately)
The accuracy of your model on the validation set is 0.9950

Out[79] - 

All tests passed

Out[81] - 

The sentiment of the sentence 
***
"It's such a nice day, I think I'll be taking Sid to Ramsgate for lunch and then to the beach maybe."
***
is positive.

The sentiment of the sentence 
***
"I hated my day, it was the worst, I'm so sad."
***
is negative.

Out[83] - 

(9088, 256)
<br/><br/>
</pre>
Out[110] - 
![image](https://user-images.githubusercontent.com/100334984/229381393-7fe822cb-ea6d-4c08-8012-135ec248e995.png)
