from .loss_jocor import loss_jocor
from .loss_structrue import loss_structrue, loss_structrue_t
from .loss_coteaching import loss_coteaching
from .loss_ntxent import NTXentLoss
from .loss_other import SCELoss, GCELoss, DMILoss, CE_SR, NegEntropy
from .loss_mixup import Mixup
from .loss_GJS import JensenShannonDivergenceWeightedScaled as GJSLoss
from .loss_CR import CR_loss
from .loss_ELR import elr_loss
from .loss_ELRplus import elr_plus_loss

__all__ = ('loss_jocor', 'loss_structrue', 'loss_coteaching', 'NTXentLoss', 'elr_loss', 'elr_plus_loss'
           'SCELoss', 'GCELoss', 'DMILoss', 'CE_SR', 'Mixup', 'GJSLoss', 'loss_CR')