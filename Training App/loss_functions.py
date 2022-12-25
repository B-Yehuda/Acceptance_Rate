from sklearn.metrics import mean_squared_error, average_precision_score, log_loss, fbeta_score


# LOSS FUNCTIONS #

class AveragePrecisionScore:
    def __init__(self, y_test, direction):
        self.y_test = y_test
        self.direction = direction
        self.name = "average_precision_score"

    def score(self, y_pred):
        return average_precision_score(self.y_test, y_pred)


class F1Score:
    def __init__(self, y_test, beta_value, direction):
        self.y_test = y_test
        self.beta_value = 1.0
        self.direction = direction
        self.name = "f1_score"

    def score(self, y_pred):
        return fbeta_score(self.y_test, y_pred, beta=self.beta_value)


class RecallScore:
    def __init__(self, y_test, beta_value, direction):
        self.y_test = y_test
        self.beta_value = 2.0
        self.direction = direction
        self.name = "recall_score"

    def score(self, y_pred):
        return fbeta_score(self.y_test, y_pred, beta=self.beta_value)


class PrecisionScore:
    def __init__(self, y_test, beta_value, direction):
        self.y_test = y_test
        self.beta_value = 0.5
        self.direction = direction
        self.name = "precision_score"

    def score(self, y_pred):
        return fbeta_score(self.y_test, y_pred, beta=self.beta_value)


class LogLoss:
    def __init__(self, y_test, direction):
        self.y_test = y_test
        self.direction = direction
        self.name = "log_loss"

    def score(self, y_pred):
        return log_loss(self.y_test, y_pred)


class RMSE:
    def __init__(self, y_test, squared, direction):
        self.y_test = y_test
        self.squared = False
        self.direction = direction
        self.name = "mean_squared_error"

    def score(self, y_pred):
        return mean_squared_error(self.y_test, y_pred, squared=self.squared)
