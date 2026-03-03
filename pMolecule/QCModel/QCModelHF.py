"""pDynamo's in-built HF QC model."""

from   pScientific.Arrays             import Array       , \
                                             StorageType
from   pScientific.LinearAlgebra      import MatrixPower
from  .GaussianBases                  import GaussianBasisContainer         , \
                                             GaussianBasisIntegralEvaluator
from  .QCModelBase                    import QCModelBase
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

    # . Defaults.
    defaultKeys  = dict ( QCModelBase.defaultKeys )
    defaultKeys.update ( { "Integral Evaluator" : ( "integralEvaluator", GaussianBasisIntegralEvaluator , None ) ,
                           "Maximum Memory"     : ( "maximumMemory"    , _DefaultMaximumMemory          , None ) ,
                           "Orbital Basis"      : ( "orbitalBasis"     , "631gs"                        , None ) } )
    defaultLabel = "HF"

    # . Charge model for properties defaults to "Mulliken".
    # . Eventually will add better option handling (e.g.using Enum) when there are more than two possible models.
    # . Is Lowdin correct? Often weird for bond orders or when have large basis sets.
    # . LowdinT is identified as Yc as the density is Xc * Po * Xc^T where Po is the desired orthogonal density.
    # . This will work no matter how Xc is calculated, as Yc is S * Xc.
    # . An alternative definition is V * Ss^(1/2) which should give the same results when symmetric orthogonalization
    # . is used only.
    def AtomicDensityBondOrders ( self, density, bOs, chargeModel = "Mulliken" ):
        """Bond orders from a density."""
        if density is not None:
            if chargeModel == "Lowdin":
                indices = self.orbitalBases.oIndices
                lowdinT = self._target.scratch.inverseOrthogonalizer
                ps      = Array.WithExtent ( lowdinT.shape[1], storageType = StorageType.Symmetric )
                density.Transform ( lowdinT, ps )
            else:
                indices = self.orbitalBasisIndices
                ps      = Array.WithShape ( density.shape )
                overlap = self._target.scratch.overlapMatrix
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

    def AtomicDensityCharges ( self, density, charges, chargeModel = "Mulliken" ):
        """Atomic charges from a density."""
        if density is not None:
            if chargeModel == "Lowdin":
                indices = self.orbitalBases.oIndices
                lowdinT = self._target.scratch.inverseOrthogonalizer
                ps      = Array.WithExtent ( lowdinT.shape[1] )
                density.DiagonalOfTransform ( lowdinT, ps )
            else:
                indices = self.orbitalBasisIndices
                ps      = Array.WithExtent ( density.shape[0] )
                overlap = self._target.scratch.overlapMatrix
                density.DiagonalOfProduct ( overlap, ps )
            for ( i, start ) in enumerate ( indices[0:-1] ):
                stop        = indices[i+1]
                charges[i] -= ps[start:stop].Sum ( )

    def BuildModel ( self, target, qcSelection = None ):
        """Build the model."""
        super ( QCModelHF, self ).BuildModel ( target, qcSelection = qcSelection )
        self.CheckMemory ( )

    def CheckMemory ( self ):
        """A simple memory check."""
        # . Determine the space required by the TEIs in memory assuming no sparsity and 64-bit real (x1) and 16-bit integer (x4) values.
        n = float ( self.orbitalBasisIndices[-1] )
        p = ( n * ( n + 1.0 ) ) / 2.0
        q = ( p * ( p + 1.0 ) ) / 2.0
        m = 16.0 * q / 1.0e+09 # . 16 bytes (1 Real64 + 4 Integer16) into GB.
        if m > self.maximumMemory:
            raise QCModelError ( "Estimated memory, {:.3f} GB, exceeds maximum memory, {:.3f} GB.".format ( m, self.maximumMemory ) )

    def EnergyClosureGradients ( self ):
        """Gradient energy closure."""
        def a ( ): self.integralEvaluator.ElectronNuclearGradients ( self._target )
        def b ( ): self.integralEvaluator.KineticOverlapGradients  ( self._target )
        def c ( ): self.integralEvaluator.TwoElectronGradients     ( self._target )
        def d ( ): self.GetWeightedDensity ( )
        return [ ( EnergyClosurePriority.QCGradients   , a ) ,
                 ( EnergyClosurePriority.QCGradients   , b ) ,
                 ( EnergyClosurePriority.QCGradients   , c ) ,
                 ( EnergyClosurePriority.QCPreGradients, d ) ]

    def EnergyClosureGradientsWithTimings ( self ):
        """Gradient energy closure."""
        def a ( ):
            tStart = self._target.cpuTimer.Current ( )
            self.integralEvaluator.ElectronNuclearGradients ( self._target )
            self._target.timings["QC Electron-Nuclear Gradients"] += ( self._target.cpuTimer.Current ( ) - tStart )
        def b ( ):
            tStart = self._target.cpuTimer.Current ( )
            self.integralEvaluator.KineticOverlapGradients ( self._target )
            self._target.timings["QC Kinetic and Overlap Gradients"] += ( self._target.cpuTimer.Current ( ) - tStart )
        def c ( ):
            tStart = self._target.cpuTimer.Current ( )
            self.integralEvaluator.TwoElectronGradients ( self._target )
            self._target.timings["QC Two-Electron Gradients"] += ( self._target.cpuTimer.Current ( ) - tStart )
        def d ( ):
            tStart = self._target.cpuTimer.Current ( )
            self.GetWeightedDensity ( )
            self._target.timings["QC Weighted Density"] += ( self._target.cpuTimer.Current ( ) - tStart )
        return [ ( EnergyClosurePriority.QCGradients   , a ) ,
                 ( EnergyClosurePriority.QCGradients   , b ) ,
                 ( EnergyClosurePriority.QCGradients   , c ) ,
                 ( EnergyClosurePriority.QCPreGradients, d ) ]

    def EnergyClosures ( self ):
        """Return energy closures."""
        def a ( ): self.integralEvaluator.CoreCoreEnergy           ( self._target ) # . With derivatives if necessary.
        def b ( ): self.integralEvaluator.ElectronNuclearIntegrals ( self._target )
        def c ( ): self.integralEvaluator.KineticOverlapIntegrals  ( self._target )
        def d ( ): self.integralEvaluator.TwoElectronIntegrals     ( self._target )
        def e ( ): self.GetOrthogonalizer ( )
        closures = super ( QCModelHF, self ).EnergyClosures ( )
        closures.extend ( [ ( EnergyClosurePriority.QCIntegrals, a ) ,
                            ( EnergyClosurePriority.QCIntegrals, b ) ,
                            ( EnergyClosurePriority.QCIntegrals, c ) ,
                            ( EnergyClosurePriority.QCIntegrals, d ) ,
                            ( EnergyClosurePriority.QCPreEnergy, e ) ] )
        closures.extend ( self.EnergyClosureGradients ( ) )
        return closures

    def EnergyClosuresWithTimings ( self ):
        """Return energy closures with timings."""
        def a ( ):
            tStart = self._target.cpuTimer.Current ( )
            self.integralEvaluator.CoreCoreEnergy ( self._target )
            self._target.timings["QC Nuclear"] += ( self._target.cpuTimer.Current ( ) - tStart )
        def b ( ):
            tStart = self._target.cpuTimer.Current ( )
            self.integralEvaluator.ElectronNuclearIntegrals ( self._target )
            self._target.timings["QC Electron-Nuclear Integrals"] += ( self._target.cpuTimer.Current ( ) - tStart )
        def c ( ):
            tStart = self._target.cpuTimer.Current ( )
            self.integralEvaluator.KineticOverlapIntegrals ( self._target )
            self._target.timings["QC Kinetic and Overlap Integrals"] += ( self._target.cpuTimer.Current ( ) - tStart )
        def d ( ):
            tStart = self._target.cpuTimer.Current ( )
            self.integralEvaluator.TwoElectronIntegrals ( self._target )
            self._target.timings["QC Two-Electron Integrals"] += ( self._target.cpuTimer.Current ( ) - tStart )
        def e ( ):
            tStart = self._target.cpuTimer.Current ( )
            self.GetOrthogonalizer ( )
            self._target.timings["QC Orthogonalizer"] += ( self._target.cpuTimer.Current ( ) - tStart )
        closures = super ( QCModelHF, self ).EnergyClosuresWithTimings ( )
        closures.extend ( [ ( EnergyClosurePriority.QCIntegrals, a ) ,
                            ( EnergyClosurePriority.QCIntegrals, b ) ,
                            ( EnergyClosurePriority.QCIntegrals, c ) ,
                            ( EnergyClosurePriority.QCIntegrals, d ) ,
                            ( EnergyClosurePriority.QCPreEnergy, e ) ] )
        closures.extend ( self.EnergyClosureGradientsWithTimings ( ) )
        return closures

    def EnergyInitialize ( self ):
        """Energy initialization."""
        super ( QCModelHF, self ).EnergyInitialize ( )
        n       = self.orbitalBasisIndices[-1]
        scratch = self._target.scratch
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

    def GetParameters ( self ):
        """Get the parameters for the model."""
        if self.label is None: self.label = "HF/{:s}".format ( self.orbitalBasis.upper ( ) )
        self.orbitalBases   = GaussianBasisContainer.FromParameterDirectory ( self.orbitalBasis, self.atomicNumbers )
        self.nuclearCharges = self.orbitalBases.nuclearCharges

    def GetWeightedDensity ( self ): 
        """Get the weighted density."""
        scratch = self._target.scratch
        if scratch.doGradients:
            wDM = scratch.weightedDensity
            for label in ( "orbitalsP", "orbitalsQ" ):
                orbitals = scratch.Get ( label, None )
                if orbitals is not None: orbitals.MakeWeightedDensity ( wDM )

    def SummaryEntries ( self ):
        """Summary entries."""
        entries = super ( QCModelHF, self ).SummaryEntries ( )
        if self._target is not None:
            entries.extend ( [ ( "Maximum Memory (GB)", "{:.3f}".format ( self.maximumMemory ) ) ,
                               ( "Orbital Basis"      ,                   self.orbitalBasis    ) ] )
        return entries

#===================================================================================================================================
# . Testing.
#===================================================================================================================================
if __name__ == "__main__" :
    pass
