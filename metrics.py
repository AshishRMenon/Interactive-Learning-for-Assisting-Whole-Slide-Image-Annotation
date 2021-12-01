
import numpy as np

def calculate_P_at_K(res, K):
    """
    Precision @ K assumes that there are at least K ground truth positives.
    Because otherwise it will never be equal to 1.
    """
    # if not (np.sum(res,axis=1) > K).all():
    #     raise ValueError("To calculate P@K, the requirement is that GTP>K,")

    res = res[:,:K]
    weights = np.arange(K, 0, -1)
    P_at_K_weighted_all = np.sum(weights*res,axis=1)/weights.sum()
    P_at_K_uniform_all = res.mean(axis=1)
    P_at_K_at_least_one_all = (res.sum(axis=1)>1).astype(float)
    P_at_K_sample_wise_sorted_indices = np.argsort(res.mean(axis=1))
    return {
            "P_at_K_weighted": P_at_K_weighted_all.mean(),
            "P_at_K_uniform": P_at_K_uniform_all.mean(),
            "P_at_K_at_least_one": P_at_K_at_least_one_all.mean(),
            "P_at_K_sample_wise": P_at_K_uniform_all,
            "P_at_K_sample_wise_sorted_indices": P_at_K_sample_wise_sorted_indices
           }


def perfect_P_at_K_matrix(res, K):
    res = res[:,:K]
    P_at_K_uniform_all = res.mean(axis=1)
    perfect_hit = (P_at_K_uniform_all==1)*1
    return perfect_hit

def calculate_perfect_P_at_K(res, K):
    res = res[:,:K]
    P_at_K_uniform_all = res.mean(axis=1)
    perfect_hit = (P_at_K_uniform_all==1)*1
    return perfect_hit.mean()
    

# def my_map(pred_labels,labels):
#     res = (pred_labels==labels.reshape(-1,1))*1
#     print(res.shape)
#     gtp = np.sum(res,axis=1)
#     a,b = res.shape
#     totals = np.tile(np.arange(b)+1,(a,1))
#     corrects = np.cumsum(res,axis=1)*res
#     precisions = corrects/totals
#     AP_all = precisions.sum(axis=1)/gtp
#     AP_all[np.isnan(AP_all)] = 0
#     MAP = AP_all.mean()  
#     return  MAP

def calculate_MAP(res,K=None):
    """
    AP means average precision at varying recall values (recall varies from 0 to 1)
    """
    if K is None:
        pass
    else:
        res = res[:,:K]
    gtp = np.sum(res,axis=1)
    a,b = res.shape
    totals = np.tile(np.arange(b)+1,(a,1))
    corrects = np.cumsum(res,axis=1)*res
    precisions = corrects/totals
    AP_all = precisions.sum(axis=1)/gtp
    AP_all[np.isnan(AP_all)] = 0
    return  AP_all



def calculate_MAP2(res,K=None):
    """
    AP means average precision at varying recall values (recall varies from 0 to 1)
    """
    if K is None:
        pass
    else:
        res = res[:,:K]
    a,b = res.shape
    gtp = np.sum(res,axis=1)
    corrects = np.cumsum(res,axis=1)*res
    totals = np.tile(np.arange(b)+1,(a,1))
    precisions = corrects/totals
    AP_all = precisions.sum(axis=1)/gtp
    AP_all[np.isnan(AP_all)] = 0
    MAP = AP_all.mean()
    return  MAP


def calculate_P_at_K_1D(res, K):
    """
    Precision @ K assumes that there are at least K ground truth positives.
    Because otherwise it will never be equal to 1.
    """
    if not res.sum() > K:
        raise ValueError("To calculate P@K, the requirement is that GTP>K")

    res = res[:K]
    weights = np.arange(K, 0, -1)
    P_at_K_weighted = sum(weights*res)/weights.sum()
    P_at_K_uniform = res.mean()
    P_at_K_at_least_one = 1. if res.sum()>0 else 0.
    return {
            "P_at_K_weighted": P_at_K_weighted,
            "P_at_K_uniform": P_at_K_uniform,
            "P_at_K_at_least_one": P_at_K_at_least_one,
           }


def calculate_AP_1D(res):
    """
    AP means average precision at varying recall values (recall varies from 0 to 1)
    """
    gtp = np.sum(res)
    corrects = np.cumsum(res)*res
    totals = np.arange(1,len(res)+1)
    precisions = corrects/totals
    return  1/gtp*np.sum(precisions)


def test():
    res = np.asarray([1,1,1,0,0,0,0,0,1,1,0,1,0,1])
    P_at_K = calculate_P_at_K_1D(res, 5)
    print("P@K metrics: {}".format(P_at_K))
    AP = calculate_AP_1D(res)
    print("AP = {}".format(AP))
    print("Hand-calculated numbers = P_at_K_weighted: {}, P_at_K_uniform: {}, P_at_K_at_least_one: {}, AP: {}".format(
        (5+4+3)/(5+4+3+2+1),
        3/5,
        1,
        1/7*(1/1 + 2/2 + 3/3 + 4/9 + 5/10 + 6/12 + 7/14)
    ))

    res = np.random.randint(0,2,(4000,20000))
    P_at_K = calculate_P_at_K(res, 5)
    print("P@K metrics: {}".format(P_at_K))
    MAP = calculate_MAP(res)
    print("MAP = {}".format(MAP))

if __name__=="__main__":
    test()
