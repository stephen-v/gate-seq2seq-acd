from sklearn import metrics


def compute_f1(y, pre_y, average):
    f1 = metrics.f1_score(y, pre_y, average=average, zero_division=1)
    return f1


def compute_precision(y, pre_y, average):
    f1 = metrics.precision_score(y, pre_y, average=average, zero_division=1)
    return f1


def compute_recall(y, pre_y, average):
    f1 = metrics.recall_score(y, pre_y, average=average, zero_division=1)
    return f1

