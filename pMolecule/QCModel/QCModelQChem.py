"""The QChem QC model."""

import glob, math, os, os.path, subprocess, re
import shutil
from pathlib import Path
from collections import defaultdict

from  pCore                     import logFile            , \
                                       LogFileActive      , \
                                       NotInstalledError
from  pScientific               import PeriodicTable      , \
                                       Units
from  pScientific.Arrays        import Array
from  pScientific.Geometry3     import Coordinates3       , \
                                       Vector3
from  pScientific.RandomNumbers import RandomString
from .QCDefinitions             import ChargeModel
from .QCModel                   import QCModel            , \
                                       QCModelState
from .QCModelError              import QCModelError

import cclib
import numpy as np

#===================================================================================================================================
# . Definitions.
#===================================================================================================================================
# . Default error suffix.
_DefaultErrorPrefix = "error_"

# . Default job name.
_DefaultJobName = "qchemJob"

# . Command environment variable name (resolved lazily at runtime).
_QChemCommand = "PDYNAMO3_QCHEMCOMMAND"

# . Scratch directory (uses PDYNAMO3_SCRATCH with safe fallback).
_QChemScratch = os.path.join ( os.getenv ( "PDYNAMO3_SCRATCH", "/tmp" ), "qchemTemporary" )

#===================================================================================================================================
# . State class.
#===================================================================================================================================
class QCModelQChemState ( QCModelState ):
    """State object for the QChem QC model."""

    _attributable = dict ( QCModelState._attributable )
    _attributable.update ( { "deleteJobFiles" : True ,
                             "paths"          : None } )

    def __del__ ( self ):
        """Deallocation."""
        self.DeleteJobFiles ( )

    def DeleteJobFiles ( self ):
        """Delete job files."""
        if self.deleteJobFiles:
            try:
                jobFiles = glob.glob ( self.paths["Glob"] + ".*" )
                for jobFile in jobFiles: os.remove ( jobFile )
                scratch = self.paths.get ( "Scratch", None )
                if scratch is not None: os.rmdir ( scratch )  # . Only deleted if random.
            except:
                pass

    def DeterminePaths ( self, scratch, deleteJobFiles = True, randomJob = False, randomScratch = False,
                         saveLabel = None ):
        """Determine the paths needed by a QChem job."""
        paths = {}
        if randomJob: jobName = RandomString ( )
        else:         jobName = _DefaultJobName
        if randomScratch:
            scratch          = os.path.join ( scratch, RandomString ( ) )
            paths["Scratch"] = scratch  # . Only set if random.
        jobDir = Path ( scratch )
        if saveLabel is not None:
            jobDir /= saveLabel
            # . Adjust QCSCRATCH for QChem's orbital saving, if it is already set.
            qcScratch = os.getenv ( "QCSCRATCH" )
            if qcScratch is not None:
                os.environ["QCSCRATCH"] = str ( Path ( qcScratch ) / saveLabel )
        Path.mkdir ( jobDir, parents = True, exist_ok = True )
        jobRoot       = jobDir / jobName
        paths["Glob"] = str ( jobRoot )
        for ( key, ext ) in ( ( "Input"  , "inp" ) ,
                              ( "Output" , "log" ) ,
                              ( "PC"     , "pc"  ) ):
            paths[key] = f"{jobRoot}.{ext}"
        self.deleteJobFiles = deleteJobFiles
        self.paths          = paths

    def SaveErrorFiles ( self, message ):
        """Save the input and output files for inspection if there is an error."""
        for key in ( "Input" , "Output" , "PC" ):
            path = self.paths[key]
            if os.path.exists ( path ):
                ( head, tail ) = os.path.split ( path )
                os.rename ( path, os.path.join ( head, _DefaultErrorPrefix + tail ) )
        ( head, tail ) = os.path.split ( self.paths["Glob"] )
        raise QCModelError ( message + "\nCheck the files \"{:s}*\".".format (
            os.path.join ( head, _DefaultErrorPrefix + tail ) ) )

#===================================================================================================================================
# . Model class.
#===================================================================================================================================
class QCModelQChem ( QCModel ):
    """The QChem QC model class."""

    _attributable = dict ( QCModel._attributable )
    _classLabel   = "QChem QC Model"
    _stateObject  = QCModelQChemState
    _summarizable = dict ( QCModel._summarizable )
    _attributable.update ( { "cdftChargeCons"     : None                           ,
                             "cdftChargeConsCI1"  : None                           ,
                             "cdftChargeConsCI2"  : None                           ,
                             "cdftSpinCons"       : None                           ,
                             "cdftSpinConsCI1"    : None                           ,
                             "cdftSpinConsCI2"    : None                           ,
                             "deleteJobFiles"     : True                           ,
                             "doCDFT"             : False                          ,
                             "doGSM"              : False                          ,
                             "doMinimization"     : False                          ,
                             "keywords"           : [ "HF", "6-31G*", "TIGHTSCF" ] ,
                             "numThreads"         : 1                              ,
                             "randomJob"          : False                          ,
                             "randomScratch"      : False                          ,
                             "saveLabel"          : None                           ,
                             "scratch"            : _QChemScratch                  } )
    _summarizable.update ( { "deleteJobFiles"     : "Delete Job Files"             ,
                             "numThreads"         : "Number Of Threads"            ,
                             "randomJob"          : "Random Job"                   ,
                             "randomScratch"      : "Random Scratch"               } )

    def AtomicCharges ( self, target, chargeModel = ChargeModel.Mulliken ):
        """Atomic charges."""
        source = target.scratch.qchemOutputData
        if   chargeModel is ChargeModel.Loewdin: return source.get ( "Loewdin Charges" , None )
        else:                                    return source.get ( "Mulliken Charges", None )

    def AtomicSpins ( self, target, chargeModel = ChargeModel.Mulliken ):
        """Atomic spins."""
        source = target.scratch.qchemOutputData
        if chargeModel is ChargeModel.Loewdin: return source.get ( "Loewdin Spins" , None )
        else:                                  return source.get ( "Mulliken Spins", None )

    def BondOrders ( self, target, chargeModel = None ):
        """Bond Orders - Mayer only."""
        return target.scratch.qchemOutputData.get ( "Mayer Bond Orders", None )

    def BuildModel ( self, target, qcSelection = None ):
        """Build the model."""
        state = super ( QCModelQChem, self ).BuildModel ( target, qcSelection = qcSelection )
        state.DeterminePaths ( self.scratch                         ,
                               deleteJobFiles = self.deleteJobFiles ,
                               randomJob      = self.randomJob      ,
                               randomScratch  = self.randomScratch  ,
                               saveLabel      = self.saveLabel      )
        return state

    def DipoleMoment ( self, target, center = None ):
        """Dipole Moment."""
        return target.scratch.qchemOutputData.get ( "Dipole", None )

    def Energy ( self, target ):
        """Calculate the quantum chemical energy."""
        doGradients     = target.scratch.doGradients
        qchemOutputData = {}
        state           = getattr ( target, self.__class__._stateName )
        target.scratch.qchemOutputData = qchemOutputData
        self.WriteInputFile ( target, state, doGradients, ( target.nbModel is not None ) )
        isOK = self.Execute ( state )
        if not isOK: state.SaveErrorFiles ( "Error executing program." )
        isOK = self.ReadOutputFile ( target, qchemOutputData )
        if not isOK: state.SaveErrorFiles ( "Error reading output file." )
        target.scratch.energyTerms["QChem QC"] = ( qchemOutputData["Energy"] * Units.Energy_Hartrees_To_Kilojoules_Per_Mole )

    def Execute ( self, state ):
        """Execute the job."""
        try:
            outFile = open ( state.paths["Output"], "w" )
            subprocess.check_call ( [ self.command, "-np", str ( self.numThreads ),
                                      state.paths["Input"], state.paths["Output"], "save" ],
                                    stderr = outFile, stdout = outFile )
            outFile.close ( )
            return True
        except:
            return False

    def OrbitalEnergies ( self, target ):
        """Orbital energies and HOMO and LUMO indices."""
        return ( target.scratch.qchemOutputData.get ( "Orbital Energies", None ) ,
                 target.scratch.qchemOutputData.get ( "HOMO"            , -1   ) ,
                 target.scratch.qchemOutputData.get ( "LUMO"            , -1   ) )

    def ReadLowdinCharges ( self, filename, scratch ):
        """Read Lowdin charges from a QChem output file."""
        with open ( filename ) as qchem_file:
            qchem_file_lines = qchem_file.readlines ( )
            for index, line in enumerate ( qchem_file_lines ):
                if ( "There are" in line ) and ( "shells and" in line ):
                    num_shells = list ( int ( s ) for s in line.split ( ) if s.isdigit ( ) )[0]
                if "Partial Lowdin       Populations for Occupied Alpha Orbitals" in line:
                    alpha_line_num = index
                if "Partial Lowdin        Populations for Occupied Beta Orbitals" in line:
                    beta_line_num  = index
                    break
        end_line_num  = beta_line_num + ( beta_line_num - alpha_line_num )
        header_lines  = list ( range ( alpha_line_num + 2, beta_line_num - 2, num_shells + 1 ) )
        header_lines += list ( range ( beta_line_num  + 2, end_line_num  - 2, num_shells + 1 ) )
        ranges        = [ ( header_lines[i] + 1, header_lines[i] + num_shells + 1 ) for i in range ( len ( header_lines ) ) ]
        blocks        = np.array ( [ [ line.split ( )[4:] for line in qchem_file_lines[block[0]:block[1]] ]
                                     for block in ranges ], dtype = float )
        populations   = np.concatenate ( blocks, axis = 1 )
        atom_instances   = [ tuple ( line.split ( )[1:3] ) for line in qchem_file_lines[ranges[0][0]:ranges[0][1]] ]
        atom_populations = defaultdict ( float )
        for index, atom_instance in enumerate ( atom_instances ):
            atom_populations[atom_instance] += sum ( populations[index] )
        atomic_numbers = { atom: PeriodicTable.AtomicNumber ( atom[0] ) for atom in atom_populations }
        lowdin_charges = Array.WithExtent ( len ( list ( atom_populations ) ) )
        for index, ( atom, population ) in enumerate ( atom_populations.items ( ) ):
            lowdin_charges[index] = atomic_numbers[atom] - population
        scratch["Loewdin Charges"] = lowdin_charges

    def ReadOutputFile ( self, target, qchemOutputData ):
        """Read a QChem output file. Returns True if successful, False on error."""
        state = getattr ( target, self.__class__._stateName )
        try:
            n        = len ( state.atomicNumbers )
            filename = state.paths["Output"]

            # . cclib reads the QChem output file into a structured object.
            data = cclib.io.ccread ( filename )

            # . Guard against incomplete Q-Chem output (e.g. OOM-killed before any SCF).
            if data is None or not hasattr ( data, "scfenergies" ) or len ( data.scfenergies ) == 0:
                raise AttributeError ( "Q-Chem output contains no SCF energies - job likely failed before SCF." )

            # . Assume converged if cclib does not raise an exception.
            scratch = { "Is Converged" : True }

            # . Energy: account for nuclear repulsion (ENUC) in QM/MM calculations.
            if target.scratch.doGradients:
                scratch["Energy"] = ( data.scfenergies[-1] / Units.Energy_Hartrees_To_Electron_Volts )
                with open ( filename ) as fn:
                    for line in fn:
                        if "Etot" in line:
                            lsplit = line.split ( )
                            ENUC   = 2 * float ( lsplit[-2] )
                            scratch["Energy"] = ( ( data.scfenergies[-1] / Units.Energy_Hartrees_To_Electron_Volts ) - ENUC )
                            break
            elif self.cdftChargeConsCI1:
                with open ( filename ) as fn:
                    for line in fn:
                        if "E(Tot)" in line:
                            lsplit = line.split ( )
                            E_Tot  = float ( lsplit[-1] )
                            ENUC   = 2 * float ( lsplit[-3] )
                            scratch["Energy"] = E_Tot - ENUC
                            break
            else:
                scratch["Energy"] = ( data.scfenergies[-1] / Units.Energy_Hartrees_To_Electron_Volts )

            # . CDFT-CI: extract per-state energies and electronic coupling.
            if self.cdftChargeConsCI1:
                CI_out = open ( "ResultsCI.txt", "w" )
                with open ( filename ) as fn:
                    copy = False
                    for line in fn:
                        if   line.startswith ( " CDFT-CI using one" ): copy = True
                        elif line.startswith ( " ---" ):               copy = False
                        elif copy:                                     CI_out.write ( line )
                CI_out.close ( )
                E_1 = E_2 = None
                with open ( filename ) as fn:
                    for line in fn:
                        if "CDFT-CI Energy state 1" in line:
                            E_1 = float ( line.split ( )[-1] )
                        elif "CDFT-CI Energy state 2" in line:
                            E_2 = float ( line.split ( )[-1] )
                            scratch["Energy"] = E_2
                if LogFileActive ( logFile ) and E_1 is not None and E_2 is not None:
                    logFile.Paragraph ( "=============CDFT-CI Output=============\n"
                        + "| State 1 Energy   = {:>13.6f} Eh |\n".format ( E_1 )
                        + "| State 2 Energy   = {:>13.6f} Eh |\n".format ( E_2 )
                        + "|                                      |" )
                with open ( filename ) as fn:
                    lines = fn.readlines ( )
                    for i, line in enumerate ( lines ):
                        if line.startswith ( " CDFT-CI Hamiltonian matrix in orthogonalized" ):
                            H_ab = float ( lines[i+2].split ( )[-1] )
                            if LogFileActive ( logFile ):
                                logFile.Paragraph ( "| Coupling (H_ab)  = {:>13.6f} Eh |\n"
                                    "========================================".format ( H_ab ) )
                            break

            # . Mulliken charges (skip for CDFT-CI).
            if not self.cdftChargeConsCI1:
                mulliken = Array.WithExtent ( n )
                if n == 1: mulliken[0] = data.charge
                else:
                    for i in range ( n ):
                        mulliken[i] = data.atomcharges["mulliken"][i]
                scratch["Mulliken Charges"] = mulliken

            # . Uncomment to enable Lowdin charges.
            # self.ReadLowdinCharges ( filename, scratch )

            # . Gradients (in atomic units).
            if target.scratch.doGradients:
                gradient   = target.scratch.qcGradients3AU
                cclib_grad = data.grads[-1]
                for i in range ( cclib_grad.shape[0] ):
                    for j in range ( cclib_grad.shape[1] ):
                        gradient[i,j] = cclib_grad[i,j]

            qchemOutputData.update ( scratch )
            return True
        except Exception as e:
            if LogFileActive ( logFile ):
                logFile.Paragraph ( "Exception in QCModelQChem.ReadOutputFile: {:s}".format ( str ( e ) ) )
            return False

    # . Coordinates are written in Angstroms (QChem default).
    def WriteInputFile ( self, target, state, doGradients, doQCMM ):
        """Write a QChem input file."""
        qcScratch = os.getenv ( "QCSCRATCH" )
        with open ( state.paths["Input"], "w" ) as inFile:

            # . $rem section.
            inFile.write ( "$rem\n" )
            if doGradients: inFile.write ( " JOBTYPE              FORCE" )
            else:           inFile.write ( " JOBTYPE              SP"    )

            # . Read MOs from saved checkpoint if available.
            if ( self.doGSM or self.doMinimization or self.doCDFT ) and qcScratch is not None:
                if Path ( os.path.join ( qcScratch, "save", "53.0" ) ).exists ( ):
                    inFile.write ( "\n SCF_GUESS            READ" )

            # . User-defined keywords.
            inFile.write ( "\n" + " ".join ( self.keywords ) )

            # . CDFT rem keywords.
            saved_orbitals_exist   = ( qcScratch is not None and
                                       Path ( os.path.join ( qcScratch, "save", "53.0" ) ).exists ( ) )
            cdft_constraints_exist = any ( ( self.cdftSpinCons, self.cdftChargeCons, self.cdftChargeConsCI1 ) )
            do_cdft = cdft_constraints_exist and saved_orbitals_exist

            if do_cdft or self.doCDFT:
                inFile.write ( "\n CDFT                 TRUE" )
            if self.cdftChargeConsCI1:
                inFile.write ( "\n CDFTCI               TRUE" )

            inFile.write ( "\n$end\n\n" )

            # . Helper: write a single CDFT constraint line.
            def write_cdft_constraint ( atom_range, range_coefficient, constraint_type ):
                first_atom_index = list ( state.pureQCAtoms ).index ( atom_range.start ) + 1
                last_atom_index  = first_atom_index + len ( list ( atom_range ) ) - 1
                inFile.write ( f"{range_coefficient:5d} {first_atom_index:4d} {last_atom_index:4d} {constraint_type}\n" )
                for i, ( _boundary, qc_atom_index, _distance ) in enumerate ( state.baQCPartners ):
                    if qc_atom_index in atom_range:
                        link_atom_index = len ( state.pureQCAtoms ) + i + 1
                        inFile.write ( f"{range_coefficient:5d} {link_atom_index:4d} {link_atom_index:4d} {constraint_type}\n" )

            def write_cdft_constraint_section ( constraints, constraint_type ):
                for constraint_value, atom_ranges in constraints:
                    inFile.write ( "  " + str ( constraint_value ) + "\n" )
                    for atom_range, range_coefficient in atom_ranges.items ( ):
                        write_cdft_constraint ( atom_range, range_coefficient, constraint_type )

            def write_cdft_section ( ):
                inFile.write ( "$cdft\n" )
                if self.cdftSpinCons:
                    write_cdft_constraint_section ( self.cdftSpinCons, "s" )
                if self.cdftChargeCons:
                    write_cdft_constraint_section ( self.cdftChargeCons, "" )
                if self.cdftChargeConsCI1:
                    write_cdft_constraint_section ( self.cdftChargeConsCI1, "" )
                    write_cdft_constraint_section ( self.cdftSpinConsCI1,   "s" )
                    inFile.write ( " -----------------\n" )
                    write_cdft_constraint_section ( self.cdftChargeConsCI2, "" )
                    write_cdft_constraint_section ( self.cdftSpinConsCI2,   "s" )
                inFile.write ( "$end\n\n" )

            if do_cdft or self.doCDFT:
                write_cdft_section ( )

            # . QM/MM sections.
            if doQCMM:
                inFile.write ( "$qm_atoms\n 1:{:d}\n$end\n\n".format ( len ( state.atomicNumbers ) ) )

                pc_file     = np.loadtxt ( state.paths["PC"], skiprows = 1, usecols = [ 1, 2, 3, 4 ] )
                pc_elements = np.loadtxt ( state.paths["PC"], skiprows = 1, usecols = [0], dtype = str )
                pc_charges  = pc_file[:,0]
                pc_coords   = pc_file[:,1:]
                unique_charges        = np.unique ( pc_charges )
                n_types               = 1 + len ( unique_charges )  # 1 zero-charge QM type + one per unique MM charge
                inFile.write ( "$force_field_params\n NumAtomTypes {:d}\n".format ( n_types ) )
                atom_types            = np.arange ( -1, -2 - len ( unique_charges ), step = -1 )
                atom_type_charges     = np.array ( [0.0] + list ( unique_charges ) )
                charges_to_atom_types = { atom_type_charges[i]: atom_types[i] for i in range ( 1, len ( atom_types ) ) }
                label_col  = np.full_like ( atom_types, " AtomType", dtype = np.object_ )
                data_array = np.column_stack ( ( label_col, atom_types, atom_type_charges,
                                                 np.ones_like ( atom_types ), np.zeros_like ( atom_types ) ) )
                np.savetxt ( inFile, data_array, fmt = "%s" )
                inFile.write ( "$end\n\n" )

            # . $molecule section (coordinates in Angstroms).
            qcCoordinates3 = target.scratch.qcCoordinates3
            inFile.write ( "$molecule\n" )
            if doQCMM:
                # . Q-Chem validates the charge/multiplicity parity against ALL atoms in $molecule
                # . (QM + MM background). Use the full system charge = QM charge + MM background charge
                # . so that the electron count parity matches the specified multiplicity.
                mm_background_charge = int ( round ( float ( np.sum ( pc_charges ) ) ) )
                full_system_charge   = target.electronicState.charge + mm_background_charge
                inFile.write ( "{:d} {:d}\n".format ( full_system_charge,
                                                      target.electronicState.multiplicity ) )
            else:
                inFile.write ( "{:d} {:d}\n".format ( target.electronicState.charge,
                                                      target.electronicState.multiplicity ) )
            atom_type = -1  # type -1 is the zero-charge QM placeholder in $force_field_params
            for ( i, n ) in enumerate ( state.atomicNumbers ):
                inFile.write ( "{:<4s}".format ( PeriodicTable.Symbol ( n ) ) )
                for j in range ( 3 ):
                    inFile.write ( "{:20.10f}".format ( qcCoordinates3[i,j] ) )
                if doQCMM:
                    inFile.write ( " {:d} 0 0 0 0".format ( atom_type ) )
                inFile.write ( "\n" )
            if doQCMM:
                for i in range ( pc_coords.shape[0] ):
                    inFile.write ( "{:<4s}".format ( pc_elements[i] ) )
                    for j in range ( 3 ):
                        inFile.write ( "{:20.10f}".format ( pc_coords[i,j] ) )
                    atom_type = charges_to_atom_types[pc_charges[i]]
                    inFile.write ( " {:d} 0 0 0 0\n".format ( atom_type ) )
            inFile.write ( "$end\n\n" )

    @property
    def command ( self ):
        """Get the QChem executable path (resolved lazily from the environment)."""
        command = self.__dict__.get ( "_command", None )
        if command is None:
            command = os.getenv ( _QChemCommand )
            if ( command is None ) or not ( os.path.isfile ( command ) and os.access ( command, os.X_OK ) ):
                raise NotInstalledError ( "QChem executable not found. Set {:s} to point to the qchem binary.".format (
                    _QChemCommand ) )
            else:
                self.__dict__["_command"] = command
        return command

#===================================================================================================================================
# . Testing.
#===================================================================================================================================
if __name__ == "__main__" :
    pass
