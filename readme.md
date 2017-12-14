For usage instructions, and examples, run:

    python src/depModel.py --help
    
Before running, ensure all four vocab files for training data have been generated according to the instructions in the homework. 
By default, these four files (vocabs.actions vocabs.labels vocabs.pos vocabs.word) are in the data/ direcctory. To change which folder the program looks in, set the path with the --vocab flag.

### Part 1

    Unlabeled attachment score 83.95
    Labeled attachment score 80.76

### Part 2

    Unlabeled attachment score 83.62
    Labeled attachment score 80.41
    
These results are not too different from the previous results. It seems this many more neurons in the two hidden layers result in less data compression, and therefore each neuron contributing less information to the following layer. 
### Part 3
#### Attempt 1

First, I trained the neural network part3_hid100.nnet, which was just like part 2 but used hidden layer dimensions of h1=100 and h2=100. The results were:

    Unlabeled attachment score 83.4
    Labeled attachment score 80.08
    
The score is wildly different from either the above cases, but shows that the optimal hidden layer size (assuming the dimension of the layers are the same), is somewhere between 100 and 400. So far the size that has had the best attachment score is from Part 1: 200.
#### Attempt 2
Next, I wanted to see how changing the size of the embedding dimensionality would affect the performance of the model. I reduced the size of the word embedding to 50, and the pos / dependency embeddings to 30. I used 200 as the size of the hidden layer. The results were still not as good as part 1:

    Unlabeled attachment score 83.76
    Labeled attachment score 80.64

I am assuming that these vectors were not big enough to capture all of the information that the network was able to collect about word, pos, and dependency label embeddings.
#### Attempt 3
Finally, I wanted to mess with the algorithms, programatically changing from the AdamTrainer to the MomentumSGDTrainer, from rectify to tanh, and reverting to the parameters from part 1. This did not go well:

    Unlabeled attachment score 73.04
    Labeled attachment score 66.45

