import torch


class CosineLoss(torch.nn.Module):
    def __init__(self, is_obfuscation):
        super(CosineLoss, self).__init__()
        self.is_obfuscation = is_obfuscation

    def cos_simi(self, emb_1, emb_2):
        return torch.mean(torch.sum(torch.mul(emb_2, emb_1), dim=1) / emb_2.norm(dim=1) / emb_1.norm(dim=1))

    def forward(self, protected_feature, target_feature, source_feature):
        cos_loss_list = []
        for i in range(len(protected_feature)):
            if not self.is_obfuscation:
                cos_loss_list.append(
                    1 - self.cos_simi(protected_feature[i], 
                                      target_feature[i].detach()))
            else:
                imp = 1 - self.cos_simi(protected_feature[i],
                                       target_feature[i].detach())
                obf = 1 - self.cos_simi(protected_feature[i],
                                       source_feature[i].detach())
                cos_loss_list.append(imp-obf)
        return torch.sum(torch.stack(cos_loss_list))
