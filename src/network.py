import dynet as dynet
import matplotlib.pyplot as plot
import random


class NetProperties:
    def __init__(self, word_embed_dim=64, pos_embed_dim=32, dep_embed_dim=32, h1_dim=200, h2_dim=200,
                 minibatch_size=1000, training_epochs=7, transfer_f=dynet.rectify, training_f=dynet.AdamTrainer):

        # Initialize network with  default values
        self.word_embed_dim = word_embed_dim
        self.transfer_f = transfer_f
        self.pos_embed_dim = pos_embed_dim
        self.dep_embed_dim = dep_embed_dim
        self.h1_dim = h1_dim
        self.h2_dim = h2_dim
        self.minibatch_size = minibatch_size
        self.training_epochs = training_epochs

        self.transfer_f = transfer_f
        self.training_f = training_f


class NeuralNetwork:
    def __init__(self, vocab, properties):
        self.vocab = vocab
        self.properties = properties

        # first initialize a computation graph container (or model).
        self.model = dynet.Model()

        # create embeddings for word, tag, and dependency label features
        self.word_embedding = self.model.add_lookup_parameters((vocab.word_size(), properties.word_embed_dim))
        self.pos_embedding = self.model.add_lookup_parameters((vocab.pos_size(), properties.pos_embed_dim))
        self.dep_embedding = self.model.add_lookup_parameters((vocab.labels_size(), properties.dep_embed_dim))

        # Assign the training and transfer functions as defined in the network parameters
        self.transfer = properties.transfer_f
        self.updater = properties.training_f(self.model)

        # Define the input dimension to the  embedding layer
        self.input_dim = 20 * properties.word_embed_dim + 20 * properties.pos_embed_dim + 12 * properties.dep_embed_dim

        # Define the first hidden layer
        self.hidden_layer_1 = self.model.add_parameters((properties.h1_dim, self.input_dim))

        # define the first hidden layer bias term and initialize it as constant 0.2.
        self.hl1_bias = self.model.add_parameters(properties.h1_dim, init=dynet.ConstInitializer(0.2))

        # Define the second hidden layer
        self.hidden_layer_2 = self.model.add_parameters((properties.h2_dim, properties.h1_dim))

        # Define the second hidden layer bias term and initialize it as constant 0.2.
        self.hl2_bias = self.model.add_parameters(properties.h2_dim, init=dynet.ConstInitializer(0.2))

        # define the output weight.
        self.output_layer = self.model.add_parameters((vocab.actions_size(), properties.h2_dim))

        # define the bias vector and initialize it as zero.
        self.output_bias = self.model.add_parameters(vocab.actions_size(), init=dynet.ConstInitializer(0))

    def forward(self, features):
        """
        Build computational graph.
        :param features:
        :return:
        """

        # Concatenate feature vectors
        embedding_layer = self.vectorize(features)

        h1 = self.transfer(self.hidden_layer_1.expr() * embedding_layer + self.hl1_bias.expr())

        h2 = self.transfer(self.hidden_layer_2.expr() * h1 + self.hl2_bias.expr())

        output = self.output_layer.expr() * h2 + self.output_bias.expr()

        return output

    def vectorize(self, features):
        """
        Compute numberical feature vector from feature list.
        :param features:
        :return:
        """

        # Extract feature id's
        word_ids = [self.vocab.word2id(word_feat) for word_feat in features[0:20]]
        pos_ids = [self.vocab.pos2id(pos_feat) for pos_feat in features[20:40]]
        dep_ids = [self.vocab.label2id(dep_feat) for dep_feat in features[40:]]

        # Extract feature embeddings
        word_feature_embeds = [self.word_embedding[wid] for wid in word_ids]
        pos_feature_embeds = [self.pos_embedding[posid] for posid in pos_ids]
        dep_feature_embeds = [self.dep_embedding[depid] for depid in dep_ids]

        # Concatenate feature vectors
        feature_vector = dynet.concatenate(word_feature_embeds + pos_feature_embeds + dep_feature_embeds)

        return feature_vector


    def train(self, train_file):
        # matplotlib configuration
        loss_values = []
        plot.ion()
        ax = plot.gca()
        ax.set_xlim([0, 10])
        ax.set_ylim([0, 3])
        plot.title("Loss over time")
        plot.xlabel("Minibatch")
        plot.ylabel("Loss")

        for epoch in range(self.properties.training_epochs):
            print 'started epoch', (epoch + 1)
            losses = []
            train_data = open(train_file, 'r').read().strip().split('\n')

            # Shuffle the training data
            random.shuffle(train_data)

            step = 0
            for line in train_data:
                fields = line.strip().split(' ')
                features, label = fields[:-1], fields[-1]
                gold_label = self.vocab.action_id[label]
                result = self.forward(features)

                # Get the loss with respect to negative log softmax function and the gold label
                loss = dynet.pickneglogsoftmax(result, gold_label)

                # Append to the minibatch losses
                losses.append(loss)
                step += 1

                if len(losses) >= self.properties.minibatch_size:
                    # We have enough loss values to compute minibatch loss
                    minibatch_loss = dynet.esum(losses) / len(losses)

                    # Run dynet's forward computation for all minibatch items, get float value
                    minibatch_loss.forward()
                    minibatch_loss_value = minibatch_loss.value()

                    # Printing / Plotting info
                    loss_values.append(minibatch_loss_value)
                    if len(loss_values) % 10 == 0:
                        ax.set_xlim([0, len(loss_values) + 10])
                        ax.plot(loss_values)
                        plot.draw()
                        plot.pause(0.0001)
                        progress = round(100 * float(step) / len(train_data), 2)
                        total_progress = round(100 * float(epoch * len(train_data) + step) / len(train_data * self.properties.training_epochs), 2)
                        print 'current minibatch loss', minibatch_loss_value, 'epoch_progress:', progress, '%', 'total', total_progress, '%'

                    # Call dynet to run backpropogation
                    minibatch_loss.backward()

                    # Call dynet to change parameter values w/ respect to current backpropogation
                    self.updater.update()

                    # Empty the loss vector
                    losses = []

                    # Refress dynet's memory
                    dynet.renew_cg()

            # Apparently, there are still some minibatch items in the memory but they are smaller than the minibatch
            # size so we ask dynet to forget them
            dynet.renew_cg()

    def score(self, features):

        # Run vector through neural network
        output = self.forward(features)

        scores = output.npvalue()

        return scores

    def load(self, filename):
        self.model.populate(filename)

    def save(self, filename):
        self.model.save(filename)





