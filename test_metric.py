from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import sklearn as sk

def print_result(y_true, y_pred):
    print("accuracy", sk.metrics.accuracy_score(y_true, y_pred))
    print("Precision", sk.metrics.precision_score(y_true, y_pred))
    print( "Recall", sk.metrics.recall_score(y_true, y_pred))
    print( "f1_score", sk.metrics.f1_score(y_true, y_pred))
    print( "confusion_matrix")
    print( sk.metrics.confusion_matrix(y_true, y_pred))
    fpr, tpr, tresholds = sk.metrics.roc_curve(y_true, y_pred)
    print("fpr,tpr,tresholds:",fpr,tpr,tresholds)
