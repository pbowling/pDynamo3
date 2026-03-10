"""Defines the QC/MM electrostatic model appropriate for QChem."""

from  pCore                  import logFile                , \
                                    LogFileActive
from  pScientific            import Units, PeriodicTable
from .QCMMElectrostaticModel import QCMMElectrostaticModel

import cclib
import numpy as np
from pathlib import Path

#===================================================================================================================================
# . Class.
#===================================================================================================================================
class QCMMElectrostaticModelQChem ( QCMMElectrostaticModel ):

    # . Defaults.
    _classLabel  = "QChem QC/MM Electrostatic Model"

    def QCMMGradients ( self, target ):
        """Read MM data from QChem output file in atomic units and convert to pDynamo units."""
        if target.scratch.doGradients:
            gradients3B    = target.scratch.Get ( "bpGradients3", None )
            gradients3M    = target.scratch.gradients3
            mmAtoms        = target.mmState.pureMMAtoms
            nM             = len ( mmAtoms )
            qchem_log_path = target.qcState.paths["Output"]
            with open ( qchem_log_path ) as qchem_log_file:
                for num, line in enumerate ( qchem_log_file ):
                    if "Charge scaled E field on MM atoms, another gradient component" in line:
                        pc_gradients = np.loadtxt ( qchem_log_path, skiprows = num + 4,
                                                    usecols = ( 1, 2, 3 ), max_rows = nM )
                        break
            # . Convert from Eh/Bohr to kJ/mol/Angstrom.
            pc_gradients *= Units.Energy_Hartrees_To_Kilojoules_Per_Mole
            for i, row in enumerate ( pc_gradients ):
                if i < nM:
                    s = mmAtoms[i]
                    for j in range ( 3 ):
                        gradients3M[s,j] += row[j]
                else:
                    # . Boundary charges.
                    s = i - nM
                    for j in range ( 3 ):
                        gradients3B[s,j] += row[j]

    def QCMMPotentials ( self, target ):
        """Write MM data to an external point-charge file (coordinates in Angstroms)."""
        outPath = target.qcState.paths.get ( "PC", None )
        if outPath is not None:
            state         = getattr ( target, self.__class__._stateName )
            chargesB      = state.bpCharges
            chargesM      = target.mmState.charges
            coordinates3B = target.scratch.Get ( "bpCoordinates3", None                )
            coordinates3M = target.scratch.Get ( "coordinates3NB", target.coordinates3 )
            mmAtoms       = target.mmState.pureMMAtoms
            qScale        = 1.0 / self.dielectric
            nM            = len ( mmAtoms )
            if chargesB is None: nB = 0
            else:                nB = len ( chargesB )
            if ( nB + nM ) > 0:
                pcFile = open ( outPath, "w" )
                pcFile.write ( "{:10d}\n".format ( nB + nM ) )
                for i in mmAtoms:
                    atom = target.atoms[i]
                    pcFile.write ( "{:>2s}{:10.5f}".format ( PeriodicTable.Symbol ( atom.atomicNumber ),
                                                              qScale * chargesM[i] ) )
                    for j in range ( 3 ):
                        pcFile.write ( "{:20.10f}".format ( coordinates3M[i,j] ) )
                    pcFile.write ( "\n" )
                for i in range ( nB ):
                    # . Boundary charges (use hydrogen symbol as placeholder).
                    pcFile.write ( " H{:10.5f}".format ( qScale * chargesB[i] ) )
                    for j in range ( 3 ):
                        pcFile.write ( "{:20.10f}".format ( coordinates3B[i,j] ) )
                    pcFile.write ( "\n" )
#===================================================================================================================================
# . Testing.
#===================================================================================================================================
if __name__ == "__main__" :
    pass
