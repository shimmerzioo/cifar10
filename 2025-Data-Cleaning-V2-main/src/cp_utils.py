import numpy as np


def classifier_logits_ncscore(logits, labels):
    return 1 - logits.softmax(dim=1)[np.arange(len(labels)), labels]


def classifier_cumulative_ncscore(logits, labels):
    logits = logits.softmax(dim=1)
    gt = logits[np.arange(len(labels)), labels].view(len(labels), 1)
    return (logits * (logits >= gt)).sum(1)


CLF_NCSCORE_MAP = {
    "CLF-Logits": classifier_logits_ncscore,
    "CLF-Cumulative": classifier_cumulative_ncscore,
}
