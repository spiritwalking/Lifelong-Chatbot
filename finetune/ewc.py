import torch
import torch.nn as nn


class EWC(object):
    def __init__(self, model, dataloaders):
        self.model = model
        self.dataloaders = dataloaders

        self.params = {n: p for n, p in self.model.named_parameters() if
                       p.requires_grad}  # extract all parameters in models
        self.p_old = {}  # initialize parameters
        self._precision_matrices = self._calculate_importance()  # generate Fisher (F) matrix for EWC

        for n, p in self.params.items():
            self.p_old[n] = p.clone().detach()  # keep the old parameter in self.p_old

    def _calculate_importance(self):
        precision_matrices = {}
        for n, p in self.params.items():
            # initialize Fisher (F) matrix（all fill zero）
            precision_matrices[n] = p.clone().detach().fill_(0)

        self.model.eval()
        if len(self.dataloaders) > 0:
            number_data = sum([len(loader) for loader in self.dataloaders])
            for dataloader in self.dataloaders:
                for batch in dataloader:
                    self.model.zero_grad()
                    for key in batch:
                        batch[key] = batch[key].cuda()
                    output = self.model(**batch)

                    ############################################################################
                    #####                     generate Fisher(F) matrix for EWC            #####
                    ############################################################################
                    loss = output.loss.mean()
                    loss.backward()
                    ############################################################################

                    for n, p in self.model.named_parameters():
                        # get the gradient of each parameter and square it, then average it in all validation set.
                        precision_matrices[n].data += p.grad.data ** 2 / number_data

            precision_matrices = {n: p for n, p in precision_matrices.items()}

        return precision_matrices

    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            # generate the final regularization term by the ewc weight (self._precision_matrices[n]) and the square of weight difference ((p - self.p_old[n]) ** 2).
            _loss = self._precision_matrices[n] * (p - self.p_old[n]) ** 2
            loss += _loss.sum()
        return loss

