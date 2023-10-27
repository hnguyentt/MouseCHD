from sklearn import metrics

########################
# Segmentation metrics #
########################
def calc_dsc(pred, gt):
    pred = (pred != 0)
    gt = (gt != 0)
    overlap = (pred & gt)
    
    return 2*overlap.sum()/(pred.sum() + gt.sum())

def calc_recall(pred, gt):
    pred = (pred != 0)
    gt = (gt != 0)
    
    return (pred & gt).sum()/gt.sum()

def calc_precision(pred, gt):
    pred = (pred != 0)
    gt = (gt != 0)
    
    tp = (pred & gt).sum()
    fp = (pred & (1-gt)).sum()
    
    return tp/(tp+fp)

def calc_accuracy(pred, gt):
    pred = (pred != 0)
    gt = (gt != 0)
    
    tp = (pred & gt).sum()
    tn = ((1 - pred) & (1 - gt)).sum()
    
    return (tp + tn)/(pred.shape[0]*pred.shape[1]*pred.shape[2])