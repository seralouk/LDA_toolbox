    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import interp
    from sklearn.datasets import make_classification
    from sklearn.cross_validation import KFold
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.model_selection import cross_val_predict
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import KFold
    from sklearn.metrics import roc_curve, auc
    from itertools import cycle
    from scipy import interp
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import label_binarize
    plt.style.use('ggplot')
    plt.figure(figsize=(12, 8))


    X, y = make_classification(n_samples=500, random_state=100, n_classes=3,n_clusters_per_class=1, flip_y=0.3)

    kf = KFold(n_splits = 5, shuffle = True, random_state= 0)
    clf = OneVsRestClassifier(LinearDiscriminantAnalysis())
    pipe= Pipeline([('scaler', StandardScaler()), ('clf', clf)])
    classes = np.unique(y)
    y_true = label_binarize(y, classes=classes)
    n_classes = y_true.shape[1]

    colors = ['darksalmon', 'gold', 'royalblue', 'mediumseagreen', 'violet']
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    fff=[]
    ttt=[]
    aucc=[]
    # Fit the model for each fold
    for i, (train, test) in enumerate(kf.split(X,y)):
        model = pipe.fit(X[train], y[train])
        y_score = model.predict_proba(X[test])

        # Compute ROC curve and ROC area for each class PER FOLD
        for j in range(n_classes):
            fpr[j], tpr[j], _ = roc_curve(y_true[test][:, j], y_score[:, j])
            roc_auc[j] = auc(fpr[j], tpr[j])

        # First aggregate all false positive rates per classe for each fold
        all_fpr = np.unique(np.concatenate([fpr[j] for j in range(n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for j in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[j], tpr[j])

        # Finally average it and compute AUC for EACH FOLD
        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        fff.append(all_fpr)
        ttt.append(mean_tpr)
        aucc.append(roc_auc["macro"])

    # Compute average across Folds
    fff = np.array(fff)
    ttt = np.array(ttt)
    aucc = np.array(aucc)

    all_fpr_folds = np.unique(np.concatenate([fff[j] for j in range(kf.get_n_splits())]))
        
    # Then interpolate all ROC curves at this points
    mean_tpr_folds = np.zeros_like(all_fpr_folds)
    for j in range(kf.get_n_splits()):
        mean_tpr_folds += interp(all_fpr_folds, fff[j], ttt[j])

    # Finally average it and compute AUC
    mean_tpr_folds /= float(kf.get_n_splits())

    #mean_mean_tpr_folds= mean_tpr_folds.mean(axis = 0)
    std = mean_tpr_folds.std(axis=0)

    tprs_upper = np.minimum(mean_tpr_folds + std, 1)
    tprs_lower = mean_tpr_folds - std

    plt.plot(all_fpr_folds, mean_tpr_folds, 'b', alpha = 0.8, label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (aucc.mean(), aucc.std()),)
    plt.fill_between(all_fpr_folds, tprs_lower, tprs_upper, color = 'blue', alpha = 0.2)
    plt.plot([0, 1], [0, 1], linestyle = '--', lw = 2, color = 'r', label = 'Luck', alpha= 0.8)
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc="lower right")
    plt.title('Receiver operating characteristic (ROC) curve')
    #plt.axes().set_aspect('equal', 'datalim')
    plt.show()

