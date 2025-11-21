class EarlyStopping:
    """
    Early stopping utility to stop training when validation loss stops improving.
    """

    def __init__(self, patience=10, min_delta=0.0):
        """
        Args:
            patience (int): Number of epochs with no improvement after which training stops.
            min_delta (float): Minimum change in the monitored value to qualify as improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def step(self, val_loss):
        """
        Update early stopping status based on validation loss.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            print(f"  EarlyStopping counter: {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
