"""Defines a full NB model compatible with the QChem program."""

from .NBModelFull                 import NBModelFull
from .QCMMElectrostaticModelQChem import QCMMElectrostaticModelQChem
from .QCMMLennardJonesModelFull   import QCMMLennardJonesModelFull

# . Notes:
#
#   Check when assign that there is a QC model and that it is QChem?

#===================================================================================================================================
# . Class.
#===================================================================================================================================
class NBModelQChem ( NBModelFull ):
    """Defines a full NB model compatible with the QChem program."""

    # . Defaults.
    defaultLabel = "QChem"

    def QCMMModels ( self ):
        """Default companion QC/MM models for the model."""
        return { "qcmmElectrostatics" : QCMMElectrostaticModelQChem ,
                 "qcmmLennard-Jones"  : QCMMLennardJonesModelFull  }

#===================================================================================================================================
# . Testing.
#===================================================================================================================================
if __name__ == "__main__" :
    pass
