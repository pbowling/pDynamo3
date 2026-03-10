"""pDynamo's in-built HF QC model."""

from   pScientific.Arrays             import Array       , \
                                             StorageType
from  .GaussianBases                  import GaussianBasisContainer         , \
                                             GaussianBasisIntegralEvaluator
from  .QCModelBase                    import QCModelBase                    , \
                                             QCModelBaseState
from  .QCModelError                   import QCModelError
from ..EnergyModel                    import EnergyClosurePriority

#===================================================================================================================================
# . Parameters.
#===================================================================================================================================
# . Maximum memory.
_DefaultMaximumMemory = 2.0 # GB.

#===================================================================================================================================
# . Class.
#===================================================================================================================================
class QCModelHF ( QCModelBase ):
    """An HF QC model."""

    _attributable = dict ( QCModelBase._attributable )
    _classLabel   = "HF QC Model"
    _stateObject  = QCModelBaseState
    _summarizable = dict ( QCModelBase._summarizable )
    _attributable.update ( { "maximumMemory" : _DefaultMaximumMemory ,
                             "orbitalBasis"  : "631gs"               } )
    _summarizable.update ( { "maximumMemory" : ( "Maximum Memory (GB)", "{:.3f}" ) ,
                             "orbitalBasis"  :   "Orbital Basis"                   } )

    # . Charge model note: Lowdin is identified as Yc (density is Xc * Po * Xc^T).
    # . This works regardless of how Xc is calculated as Yc = S * Xc.
    def AtomicDensityBondOrders ( self, target, density, bOs, chargeModel = "Mulliken" ):
        """Bond orders from a density."""
        state = getattr ( target, self.__class__._stateName )
        if density is not None:
            if chargeModel == "Lowdin":
                indices = state.orbitalBases.oIndices
                lowdinT = target.scratch.inverseOrthogonalizer
                ps      = Array.WithExtent ( lowdinT.shape[1], storageType = StorageType.Symmetric )
                density.Transform ( lowdinT, ps )
            else:
                indices = state.orbitalBases.centerFunctionPointers
                ps      = Array.WithShape ( density.shape )
                overlap = target.scratch.overlapMatrix
                density.MatrixMultiply ( overlap, ps )
            for i in range ( bOs.shape[0] ):
                uStart = indices[i  ]
                uStop  = indices[i+1]
                for j in range ( i+1 ):
                    vStart = indices[j  ]
                    vStop  = indices[j+1]
                    q      = 0.0
                    for u in range ( uStart, uStop ):
                        for v in range ( vStart, vStop ): q += ( ps[u,v] * ps[v,u] )
                    bOs[i,j] += q

    def AtomicDensityCharges ( self, target, density, charges, chargeModel = "Mulliken" ):
        """Atomic charges from a density."""
        state = getattr ( target, self.__class__._stateName )
        if density is not None:
            if chargeModel == "Lowdin":
                indices = state.orbitalBases.oIndices
                lowdinT = target.scratch.inverseOrthogonalizer
                ps      = Array.WithExtent ( lowdinT.shape[1] )
                density.DiagonalOfTransform ( lowdinT, ps )
            else:
                indices = state.orbitalBases.centerFunctionPointers
                ps      = Array.WithExtent ( density.shape[0] )
                overlap = target.scratch.overlapMatrix
                density.DiagonalOfProduct ( overlap, ps )
            for ( i, start ) in enumerate ( indices[0:-1] ):
                stop        = indices[i+1]
                charges[i] -= ps[start:stop].Sum ( )

    def BuildModel ( self, target, qcSelection = None ):
        """Build the model."""
        state = super ( QCModelHF, self ).BuildModel ( target, qcSelection = qcSelection )
        self.CheckMemory ( target )
        return state

    def CheckMemory ( self, target ):
        """A simple memory check."""
        # . Estimate space required by TEIs; no sparsity, 64-bit real + 4 x 16-bit integer = 16 bytes.
        state = getattr ( target, self.__class__._stateName )
        n = float ( len ( state.orbitalBases ) )
        p = ( n * ( n + 1.0 ) ) / 2.0
        q = ( p * ( p + 1.0 ) ) / 2.0
        m = 16.0 * q / 1.0e+09
        if m > self.maximumMemory:
            raise QCModelError ( "Estimated memory, {:.3f} GB, exceeds maximum memory, {:.3f} GB.".format ( m, self.maximumMemory ) )

    def EnergyClosureGradients ( self, target ):
        """Gradient energy closures."""
        def a ( ): self.integralEvaluator.ElectronNuclearGradients ( target )
        def b ( ): self.integralEvaluator.KineticOverlapGradients  ( target )
        def c ( ): self.integralEvaluator.TwoElectronGradients     ( target )
        def d ( ): self.GetWeightedDensity ( target )
        return [ ( EnergyClosurePriority.QCGradients   , a, "QC Electron-Nuclear Gradients"    ) ,
                 ( EnergyClosurePriority.QCGradients   , b, "QC Kinetic and Overlap Gradients" ) ,
                 ( EnergyClosurePriority.QCGradients   , c, "QC Two-Electron Gradients"        ) ,
                 ( EnergyClosurePriority.QCPreGradients, d, "QC Weighted Density"              ) ]

    def EnergyClosures ( self, target ):
        """Return energy closures."""
        def a ( ): self.integralEvaluator.CoreCoreEnergy           ( target )
        def b ( ): self.integralEvaluator.ElectronNuclearIntegrals ( target )
        def c ( ): self.integralEvaluator.KineticOverlapIntegrals  ( target )
        def d ( ): self.integralEvaluator.TwoElectronIntegrals     ( target )
        def e ( ): self.GetOrthogonalizer ( target )
        closures = super ( QCModelHF, self ).EnergyClosures ( target )
        closures.extend ( [ ( EnergyClosurePriority.QCIntegrals    , a, "QC Nuclear"                       ) ,
                            ( EnergyClosurePriority.QCIntegrals    , b, "QC Electron-Nuclear Integrals"    ) ,
                            ( EnergyClosurePriority.QCIntegrals    , c, "QC Kinetic and Overlap Integrals" ) ,
                            ( EnergyClosurePriority.QCIntegrals    , d, "QC Two-Electron Integrals"        ) ,
                            ( EnergyClosurePriority.QCOrthogonalizer, e, "QC Orthogonalizer"              ) ] )
        closures.extend ( self.EnergyClosureGradients ( target ) )
        return closures

    def EnergyInitialize ( self, target ):
        """Energy initialization."""
        super ( QCModelHF, self ).EnergyInitialize ( target )
        state   = getattr ( target, self.__class__._stateName )
        n       = len ( state.orbitalBases )
        scratch = target.scratch
        overlap = scratch.Get ( "overlapMatrix", None )
        if ( overlap is None ) or ( overlap.rows != n ):
            overlap = Array.WithExtent ( n, storageType = StorageType.Symmetric )
            scratch.overlapMatrix = overlap
        overlap.Set ( 0.0 )
        if scratch.doGradients:
            wDM = scratch.Get ( "weightedDensity", None )
            if wDM is None:
                wDM = Array.WithExtent ( n, storageType = StorageType.Symmetric )
                scratch.weightedDensity = wDM
            wDM.Set ( 0.0 )

    def GetParameters ( self, target ):
        """Get the parameters for the model."""
        state                = getattr ( target, self.__class__._stateName )
        state.orbitalBases   = GaussianBasisContainer.FromParameterDirectory ( self.orbitalBasis, state.atomicNumbers )
        state.nuclearCharges = state.orbitalBases.nuclearCharges

    def GetWeightedDensity ( self, target ):
        """Get the weighted density."""
        scratch = target.scratch
        if scratch.doGradients:
            wDM = scratch.weightedDensity
            for label in ( "orbitalsP", "orbitalsQ" ):
                orbitals = scratch.Get ( label, None )
                if orbitals is not None: orbitals.MakeWeightedDensity ( wDM )

#===================================================================================================================================
# . Testing.
#===================================================================================================================================
if __name__ == "__main__" :
    pass
