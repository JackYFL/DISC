from .StandardCE import StandardCE, StandardCETest
from .Decoupling import Decoupling
from .Coteaching import Coteaching
from .Coteachingplus import Coteachingplus
from .JoCoR import JoCoR
from .Colearning import Colearning
from .DISC import DISC
from .Mixup import Mixup
from .NL import NegtiveLearning
from .MetaLearning import MetaLearning
from .JointOptimization import JointOptimization
from .PENCIL import PENCIL
from .GJS import GJS
from .ELR import ELR

__all__ = ('DISC', 'StandardCE', 'Decoupling', 'Coteaching', 'Coteachingplus', 'JoCoR', 'Mixup', 'ELR', 
           'Colearning', 'NegtiveLearning', 'JointOptimization', 'MetaLearning',
           'PENCIL', 'GJS', 'StandardCETest')
