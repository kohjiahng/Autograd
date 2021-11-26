class ModelTrainer:
    '''
    Class to handle all training/cross validations
    '''
    def __init__(self, model, optimizer, loss, metrics):
        '''
        metrics are loss (default) and accuracy 
        '''
        if 'loss' not in metrics:
            metrics.append('loss')

        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.train_log = {metric:[] for metric in metrics}
        self.test_log = {metric:[] for metric in metrics}
    def train(self, dataloader):
        '''
        Trains self.model for one epoch
        dataloader returns a tuple of (arguments, targets)
        Assumed that the last element are targets, rest of the elements are passed into self.model 
        '''
        self.model.train()
        for metric in self.metrics:
            self.train_log[metric].append(0)
        for data in dataloader:
            output = self.model(*data[:-1])
            loss = self.loss(output, data[-1])

            self.train_log['loss'][-1] += loss.item() * len(data[-1]) # Assumes loss is averaged
            if 'accuracy' in self.metrics:
                self.train_log['accuracy'][-1] += (output.no_grad().argmax(-1) == data[-1].argmax(-1)).asarray().sum()

            loss.backward()
            self.optimizer.step()
        for metric in self.metrics:
            self.train_log[metric][-1] /= len(dataloader.dataset)
    def eval(self, dataloader):
        '''
        Evaluates self.model for one epoch, stores metrics into self.test_log
        dataloader returns a tuple of (arguments, targets)
        Assumed that the last element are targets, rest of the elements are passed into self.model 
        '''
        self.model.eval()
        for metric in self.metrics:
            self.test_log[metric].append(0)
        for data in dataloader:
            output = self.model(*data[:-1])
            loss = self.loss(output, data[-1])

            self.test_log['loss'][-1] += loss.item() * len(data[-1]) # Assumes loss is averaged
            if 'accuracy' in self.metrics:
                self.test_log['accuracy'][-1] += (output.argmax(-1) == data[-1].argmax(-1)).asarray().sum()
        for metric in self.metrics:
            self.test_log[metric][-1] /= len(dataloader.dataset)