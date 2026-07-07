"""Defines a full NB model compatible with the QChem program."""

from  .NBModelError               import NBModelError
from  .NBModelFull                import NBModelFull
from  .QCMMElectrostaticModelQChem import QCMMElectrostaticModelQChem
from  .QCMMLennardJonesModelFull  import QCMMLennardJonesModelFull
from  ..EnergyModel               import EnergyModelPriority

# . Notes:
#
#   Check when assign that there is a QC model and that it is QChem?

#===================================================================================================================================
# . Class.
#===================================================================================================================================
class NBModelQChem ( NBModelFull ):
    """Defines a full NB model compatible with the QChem program."""

    # . Defaults.
    _classLabel   = "QChem NB Model"
    _attributable = dict ( NBModelFull._attributable )
    _attributable.update ( { "mmCutoff"  : 0.0 ,   # . Angstrom; 0 = include all MM atoms.
                              "mmMinDist" : 0.5 } )  # . Angstrom; minimum QM-MM distance; 0 = disabled.

    def BuildModel ( self, target, assignQCMMModels = True ):
        """Build the model, forwarding mmCutoff to the QC/MM electrostatic companion."""
        # . Call parent BuildModel but without auto-assigning QC/MM models so we can
        # . create QCMMElectrostaticModelQChem with the correct mmCutoff.
        super ( NBModelQChem, self ).BuildModel ( target, assignQCMMModels = False )
        if assignQCMMModels               and \
           ( target.qcModel is not None ) and \
           ( len ( target.qcState.qcAtoms ) <= len ( target.atoms ) ):
            withSymmetry = ( target.symmetryParameters is not None )
            for ( key, valueClass ) in self.QCMMModels ( qcModel = target.qcModel, withSymmetry = withSymmetry ).items ( ):
                if key == "qcmmElectrostatic":
                    model = valueClass.WithOptions ( mmCutoff  = self.mmCutoff  ,
                                                     mmMinDist = self.mmMinDist )
                else:
                    model = valueClass.WithDefaults ( )
                target._AddEnergyModel ( key, model, priority = EnergyModelPriority.QCMMModel )

    def QCMMModels ( self, qcModel = None, withSymmetry = False ):
        """Default companion QC/MM models for the model."""
        return { "qcmmElectrostatic" : QCMMElectrostaticModelQChem ,
                 "qcmmLennardJones"  : QCMMLennardJonesModelFull  }

#===================================================================================================================================
# . Testing.
#===================================================================================================================================
if __name__ == "__main__" :
    pass
