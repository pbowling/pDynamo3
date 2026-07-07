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
    _classLabel   = "QChem QC/MM Electrostatic Model"
    _attributable = dict ( QCMMElectrostaticModel._attributable )
    _attributable.update ( { "mmCutoff"  : 0.0 ,   # . Angstrom; 0 = include all MM atoms.
                              "mmMinDist" : 0.5 } )  # . Angstrom; minimum QM-MM distance (removes unphysical close contacts near link atoms); 0 = disabled.
    _summarizable = dict ( getattr ( QCMMElectrostaticModel, "_summarizable", {} ) )
    _summarizable.update ( { "mmCutoff"  : ( "MM Cutoff (Ang)",   "{:.1f}" ) ,
                              "mmMinDist" : ( "MM Min Dist (Ang)", "{:.1f}" ) } )

    def QCMMGradients ( self, target ):
        """Read MM data from QChem output file in atomic units and convert to pDynamo units."""
        if target.scratch.doGradients:
            gradients3B    = target.scratch.Get ( "bpGradients3", None )
            gradients3M    = target.scratch.gradients3
            # . Use the filtered MM atom list if a cutoff was applied, otherwise all pure MM atoms.
            mmAtomList     = getattr ( self, "_activeMM", None )
            if mmAtomList is None:
                mmAtomList = list ( target.mmState.pureMMAtoms )
            nM             = len ( mmAtomList )
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
                    s = mmAtomList[i]   # . actual system atom index (filtered or original)
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
            allMM         = list ( target.mmState.pureMMAtoms )
            qScale        = 1.0 / self.dielectric
            # . Apply spatial filters using ALL QC+link atom positions from qcCoordinates3QCMM.
            # . mmCutoff: max distance filter (keep MM atoms within mmCutoff of any QC/link atom).
            # . mmMinDist: min distance filter (remove MM atoms < mmMinDist from any QC/link atom;
            # .   default 0.5 Ang guards against unphysical close contacts near link atoms
            # .   that crash Q-Chem ZEOLITE interface).
            if self.mmCutoff > 0.0 or self.mmMinDist > 0.0:
                # . qcCoordinates3QCMM contains pure QC + link atom positions (Angstroms).
                qcCrd3QCMM = target.scratch.qcCoordinates3QCMM
                nQCAll     = len ( target.qcState.atomicNumbers )           # . pureQC + link atoms
                qcCoords   = np.array ( [ [ qcCrd3QCMM[i,j] for j in range ( 3 ) ]
                                           for i in range ( nQCAll ) ] )    # (nQCAll, 3) Ang
                mmCoords   = np.array ( [ coordinates3M[i] for i in allMM ] )  # (nMM, 3) Ang
                diffs      = mmCoords [ :, np.newaxis, : ] - qcCoords [ np.newaxis, :, : ]  # (nMM, nQCAll, 3)
                minDists   = np.sqrt ( np.min ( np.sum ( diffs ** 2, axis = 2 ), axis = 1 ) )  # (nMM,)
                mask       = np.ones ( len ( allMM ), dtype = bool )
                if self.mmCutoff  > 0.0: mask &= ( minDists <= self.mmCutoff  )
                if self.mmMinDist > 0.0: mask &= ( minDists >= self.mmMinDist )
                mmAtoms    = [ allMM[i] for i in range ( len ( allMM ) ) if mask[i] ]
            else:
                mmAtoms    = allMM
            self._activeMM = mmAtoms   # . saved for QCMMGradients
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
