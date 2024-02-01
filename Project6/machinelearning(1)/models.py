import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        # compute the dot product of the stored weight vector and the given input
        return nn.DotProduct(self.get_weights(),  x)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        # which should return 1 if the dot product is non-negative or -1 otherwise
        if nn.as_scalar(self.run(x) )  >= 0: 
            return 1
        else: 
            return -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        # This should repeatedly loop over the data set and make updates on examples that are misclassified
        # Use the update method of the nn.Parameter class to update the weights.
        while True:
            judge = True
            for x, y in dataset.iterate_once(1):
                if nn.as_scalar(y) == self.get_prediction(x): continue
                else:
                    self.w.update(x, nn.as_scalar(y))
                    judge = False
            # When an entire pass over the data set is completed without making any mistakes, 
            # 100% training accuracy has been achieved, and training can terminate.
            if judge: break


class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.batch_size = 8
        self.learning_rate = 0.01
        self.w1 = nn.Parameter(1, 128)
        self.b1 = nn.Parameter(1, 128)
        self.w2 = nn.Parameter(128, 1)
        self.b2 = nn.Parameter(1, 1)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        wx_1 = nn.Linear(x, self.w1)
        wx_b_1 = nn.AddBias(wx_1, self.b1)
        relu_1 = nn.ReLU(wx_b_1)
        wx_2 = nn.Linear(relu_1, self.w2)
        return nn.AddBias(wx_2, self.b2)

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SquareLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        while True:
            for x, y in dataset.iterate_once(self.batch_size):
                # the gradients of the loss with respect to the parameters:
                gradient_list = nn.gradients(self.get_loss(x, y), [self.w1, self.w2, self.b1, self.b2])
                # update our parameters
                # multiplier variable based on a suitable learning rate of our choosing
                self.w1.update(gradient_list[0], -self.learning_rate)
                self.w2.update(gradient_list[1], -self.learning_rate)
                self.b1.update(gradient_list[2], -self.learning_rate)
                self.b2.update(gradient_list[3], -self.learning_rate)
            # implementation will receive full points if it gets a loss of 0.02 or better
            # (use nn.as_scalar to convert a loss node to a Python number)
            if nn.as_scalar( self.get_loss(nn.Constant(dataset.x), nn.Constant(dataset.y)) ) < 0.02: break

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.batch_size = 25
        self.learning_rate = 0.1
        self.w1 = nn.Parameter(784, 128)
        self.w2 = nn.Parameter(128, 100)
        self.w3 = nn.Parameter(100, 10)
        self.b1 = nn.Parameter(1, 128)
        self.b2 = nn.Parameter(1, 100)
        self.b3 = nn.Parameter(1, 10)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        wx_1 = nn.Linear(x, self.w1)
        wx_b_1 = nn.AddBias(wx_1, self.b1)
        relu_1 = nn.ReLU(wx_b_1)
        wx_2 = nn.Linear(relu_1, self.w2)
        wx_b_2 = nn.AddBias(wx_2, self.b2)
        relu_2 = nn.ReLU(wx_b_2)
        wx_3 = nn.Linear(relu_2, self.w3)
        wx_b_3 = nn.AddBias(wx_3, self.b3)
        return wx_b_3

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SoftmaxLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        while True:
            for x, y in dataset.iterate_once(self.batch_size):
                # the gradients of the loss with respect to the parameters:
                gradient_list = nn.gradients(self.get_loss(x, y), [self.w1, self.w2, self.w3, self.b1, self.b2, self.b3])
                # update our parameters
                # multiplier variable based on a suitable learning rate of our choosing
                self.w1.update(gradient_list[0], -self.learning_rate)
                self.w2.update(gradient_list[1], -self.learning_rate)
                self.w3.update(gradient_list[2], -self.learning_rate)
                self.b1.update(gradient_list[3], -self.learning_rate)
                self.b2.update(gradient_list[4], -self.learning_rate)
                self.b3.update(gradient_list[5], -self.learning_rate)

            if dataset.get_validation_accuracy() > 0.98: break
        

class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.batch_size = 25
        self.learning_rate = 0.1
        self.w1 = nn.Parameter(47, 100)
        self.w2 = nn.Parameter(100, 100)
        self.w3 = nn.Parameter(100, 5)

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        # z = nn.Add(nn.Linear(x, W), nn.Linear(h, W_hidden))
        h = nn.ReLU(nn.Linear(xs[0], self.w1))
        for x in xs[1: ]:
            h = nn.ReLU(
                nn.Add(nn.Linear(x, self.w1), 
                       nn.Linear(h, self.w2))
                )
        return nn.Linear(h, self.w3)

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SoftmaxLoss(self.run(xs), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        for x, y in dataset.iterate_forever(self.batch_size):
            gradient_list = nn.gradients(self.get_loss(x, y), [self.w1, self.w2, self.w3])
            self.w1.update(gradient_list[0], -self.learning_rate)
            self.w2.update(gradient_list[1], -self.learning_rate)
            self.w3.update(gradient_list[2], -self.learning_rate)
            if dataset.get_validation_accuracy() > 0.88: break
            
