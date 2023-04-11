import torch
import torch.nn.functional as F
from train_utils import ce_loss


class Get_Scalar:
    def __init__(self, value):
        self.value = value

    def get_value(self, iter):
        return self.value

    def __call__(self, iter):
        return self.value

def causal_inference(current_logit, qhat, tau=0.5):
    # de-bias pseudo-labels
    debiased_prob = F.softmax(current_logit - tau*torch.log(qhat), dim=1)
    return debiased_prob


def consistency_loss(logits_s, logits_w, qhat, name='ce', e_cutoff=-8, use_hard_labels=True, use_marginal_loss=True, tau=0.5):
    assert name in ['ce', 'L2']
    logits_w = logits_w.detach()
    if name == 'L2':
        assert logits_w.size() == logits_s.size()
        return F.mse_loss(logits_s, logits_w, reduction='mean')

    elif name == 'L2_mask':
        pass

    elif name == 'ce':
        pseudo_label = F.softmax(logits_w, dim=1)

        max_probs, max_idx = torch.max(pseudo_label, dim=-1)

        energy = -torch.logsumexp(logits_w, dim=1)
        mask_raw = energy.le(e_cutoff)
        mask = mask_raw.float()
        select = mask_raw.long()

        if use_marginal_loss:
            delta_logits = torch.log(qhat)
            logits_s = logits_s + tau * delta_logits

        masked_loss = ce_loss(logits_s, max_idx, use_hard_labels, reduction='none') * mask

        return masked_loss.mean(), mask.mean(), select, max_idx.long(), mask_raw

    else:
        assert Exception('Not Implemented consistency_loss')
