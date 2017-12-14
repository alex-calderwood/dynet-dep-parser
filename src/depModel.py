import os, sys
from optparse import OptionParser
from decoder import *
from utils import Vocab
from network import NeuralNetwork, NetProperties


class DepModel:
    def __init__(self, vocab_path, model_path, training_path, net_prop=None):
        '''
            You can add more arguments for examples actions and model paths.
            You need to load your model here.
            actions: provides indices for actions.
            it has the same order as the data/vocabs.actions file.

            Feel free to add more arguments
        '''
        # if you prefer to have your own index for actions, change this.
        self.actions = ['SHIFT', 'LEFT-ARC:rroot', 'LEFT-ARC:cc', 'LEFT-ARC:number', 'LEFT-ARC:ccomp',
                        'LEFT-ARC:possessive', 'LEFT-ARC:prt', 'LEFT-ARC:num', 'LEFT-ARC:nsubjpass', 'LEFT-ARC:csubj',
                        'LEFT-ARC:conj', 'LEFT-ARC:dobj', 'LEFT-ARC:nn', 'LEFT-ARC:neg', 'LEFT-ARC:discourse',
                        'LEFT-ARC:mark', 'LEFT-ARC:auxpass', 'LEFT-ARC:infmod', 'LEFT-ARC:mwe', 'LEFT-ARC:advcl',
                        'LEFT-ARC:aux', 'LEFT-ARC:prep', 'LEFT-ARC:parataxis', 'LEFT-ARC:nsubj', 'LEFT-ARC:<null>',
                        'LEFT-ARC:rcmod', 'LEFT-ARC:advmod', 'LEFT-ARC:punct', 'LEFT-ARC:quantmod', 'LEFT-ARC:tmod',
                        'LEFT-ARC:acomp', 'LEFT-ARC:pcomp', 'LEFT-ARC:poss', 'LEFT-ARC:npadvmod', 'LEFT-ARC:xcomp',
                        'LEFT-ARC:cop', 'LEFT-ARC:partmod', 'LEFT-ARC:dep', 'LEFT-ARC:appos', 'LEFT-ARC:det',
                        'LEFT-ARC:amod', 'LEFT-ARC:pobj', 'LEFT-ARC:iobj', 'LEFT-ARC:expl', 'LEFT-ARC:predet',
                        'LEFT-ARC:preconj', 'LEFT-ARC:root', 'RIGHT-ARC:rroot', 'RIGHT-ARC:cc', 'RIGHT-ARC:number',
                        'RIGHT-ARC:ccomp', 'RIGHT-ARC:possessive', 'RIGHT-ARC:prt', 'RIGHT-ARC:num',
                        'RIGHT-ARC:nsubjpass', 'RIGHT-ARC:csubj', 'RIGHT-ARC:conj', 'RIGHT-ARC:dobj', 'RIGHT-ARC:nn',
                        'RIGHT-ARC:neg', 'RIGHT-ARC:discourse', 'RIGHT-ARC:mark', 'RIGHT-ARC:auxpass',
                        'RIGHT-ARC:infmod', 'RIGHT-ARC:mwe', 'RIGHT-ARC:advcl', 'RIGHT-ARC:aux', 'RIGHT-ARC:prep',
                        'RIGHT-ARC:parataxis', 'RIGHT-ARC:nsubj', 'RIGHT-ARC:<null>', 'RIGHT-ARC:rcmod',
                        'RIGHT-ARC:advmod', 'RIGHT-ARC:punct', 'RIGHT-ARC:quantmod', 'RIGHT-ARC:tmod',
                        'RIGHT-ARC:acomp', 'RIGHT-ARC:pcomp', 'RIGHT-ARC:poss', 'RIGHT-ARC:npadvmod', 'RIGHT-ARC:xcomp',
                        'RIGHT-ARC:cop', 'RIGHT-ARC:partmod', 'RIGHT-ARC:dep', 'RIGHT-ARC:appos', 'RIGHT-ARC:det',
                        'RIGHT-ARC:amod', 'RIGHT-ARC:pobj', 'RIGHT-ARC:iobj', 'RIGHT-ARC:expl', 'RIGHT-ARC:predet',
                        'RIGHT-ARC:preconj', 'RIGHT-ARC:root']
        # write your code here for additional parameters.

        # Initialize vocabulary
        try:
            vocab_path = vocab_path + "vocabs."
            actions_file = vocab_path + "actions"
            labels_file = vocab_path + "labels"
            pos_file = vocab_path + "pos"
            word_file = vocab_path + "word"
        except IOError:
            sys.stderr.write("Could not find vocab files in path: " + vocab_path)

        self.vocab = Vocab(actions_file, labels_file, pos_file, word_file)

        # If no properties specified, use defaults
        if net_prop is None:
            net_prop = NetProperties()

        # Initialize model with given properties
        self.network = NeuralNetwork(self.vocab, net_prop)

        # Load model or train a new one
        if training_path is None:
            print("Loading model from " + model_path)
            self.network.load(model_path)
        else:
            print("Training model from training data" + training_path)
            self.network.train(training_path)
            print("Neural network trained")

            print("Saving to", model_path)
            self.network.save(model_path)

    def score(self, str_features):
        '''
        :param str_features: String features
        20 first: words, next 20: pos, next 12: dependency labels.
        DO NOT ADD ANY ARGUMENTS TO THIS FUNCTION.
        :return: list of scores
        '''
        return self.network.score(str_features)


def usage(parser):
    parser.print_help()
    print
    print "Example usage (training):\n", \
        "\t python src/depModel.py --model out/nnet.model --train data/train.data"

    print "Example usage (parsing):\n", \
        "\t python src/depModel.py --model out/nnet.model --in trees/dev.conll --out trees/dev.conll.parsed"


def main():
    parser = OptionParser()
    parser.add_option("--model", dest="model_path", metavar="FILE", default=None,
                      help="path to the model for loading or saving (required)")
    parser.add_option("--in", dest="input_p", metavar="FILE", default=None,
                      help="input data to be parsed, leave unspecified if you are only training a model")
    parser.add_option("--out", dest="output_p", metavar="FILE", default="trees/out.conll",
                      help="results of parse, leave unspecified if you are only training a model")
    parser.add_option("--train", dest="training_path", metavar="FILE",
                      help="training data file, which can be generated using src/gen.py. Leave unspecified to load an existing model")
    parser.add_option("--vocab", dest="vocab_path", metavar="PATH", default="data/",
                      help="directory where vocab files are stored, defaults to `data/`")

    # Network Parameter Options
    parser.add_option("--epochs", dest="num_epochs", type="int", default=7,
                      help="number of epochs, default 7")
    parser.add_option("--minibatch", type="int", dest="minibatch", default=1000,
                      help="minibatch size, default 1000")
    parser.add_option("--h1", type="int", dest="h1_size", default=200,
                      help="h1 size, default 200")
    parser.add_option("--h2", type="int", dest="h2_size", default=200,
                      help="h2 size, default 200")

    (options, args) = parser.parse_args()

    properties = NetProperties(word_embed_dim=64, pos_embed_dim=32, dep_embed_dim=32, h1_dim=options.h1_size, h2_dim=options.h2_size,
                 minibatch_size=options.minibatch, training_epochs=options.num_epochs)

    if options.model_path is None:
        sys.stderr.write("Error: No model specified for loading or saving.\n")
        usage(parser)
        sys.exit(1)

    if options.training_path is None and options.input_p is None:
        sys.stderr.write("Error: You must specify a file to parse and/or a file to train a new model.\n")
        usage(parser)
        sys.exit(1)

    m = DepModel(options.vocab_path, options.model_path, options.training_path, net_prop=properties)

    if options.input_p is not None:
        print("Beginning parse.")
        Decoder(m.score, m.actions).parse(options.input_p, options.output_p)


if __name__ == '__main__':
    main()
