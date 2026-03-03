# File based on QCModelORCA

"""The QChem QC model."""

import glob, math, os, os.path, subprocess, re
import shutil
from pathlib import Path
from collections import defaultdict

from   pCore                 import logFile       , \
                                    LogFileActive
from   pScientific           import PeriodicTable , \
                                    Units
from   pScientific.Arrays    import Array
from   pScientific.Geometry3 import Coordinates3  , \
                                    Vector3
from  .QCModel               import QCModel
from  .QCModelError          import QCModelError

import cclib
import numpy as np

#===================================================================================================================================
# . Definitions.
#===================================================================================================================================
# . Default error suffix.
_DefaultErrorPrefix = "error_"

# . Environment variables.
_QChemCommand = os.getenv ( "PDYNAMO_QCHEMCOMMAND" )
_QChemScratch = os.path.join ( os.getenv ( "PDYNAMO_SCRATCH" ), "qchemTemporary" )
QCSCRATCH = "QCSCRATCH"

#===================================================================================================================================
# . Set handlers (and other miscellaneous functions).
#===================================================================================================================================
import random, string
def _RandomString ( characters = ( string.ascii_lowercase + string.digits ), size = 12, startingCharacters = None ):
    """Generate a random string."""
    if startingCharacters is None: startingCharacters = string.ascii_lowercase
    return "".join ( [ random.choice ( startingCharacters ) ] + [ random.choice ( characters ) for x in range ( size - 1 ) ] )

def _SetHandlerPathOption ( self, key, value ):
    """Path option set handler."""
    if not isinstance ( value, str ) or ( len ( value ) <= 0 ): raise TypeError ( "Invalid \"{:s}\" argument.".format ( key ) )
    self._entries[key] = value
    self.DeterminePaths ( )

def _SetHandlerRandomOption ( self, key, value ):
    """Random option set handler."""
    if not isinstance ( value, bool ): raise TypeError ( "Invalid \"{:s}\" argument.".format ( key ) )
    self._entries[key] = value
    self.DeterminePaths ( )

#===================================================================================================================================
# . Class.
#===================================================================================================================================
class QCModelQChem ( QCModel ):
    """The QChem QC model class."""

    # . Defaults.
    defaultKeys  = { "Command"              : ( "command"           , _QChemCommand                  , None                    ) ,
                     "Delete Job Files"     : ( "deleteJobFiles"    , True                           , None                    ) ,
                     "Job Name"             : ( "job"               , "qchemJob"                     , _SetHandlerPathOption   ) ,
                     "Keywords"             : ( "keywords"          , [ "HF", "6-31G*", "TIGHTSCF" ] , None                    ) ,
                     "Random Job Name"      : ( "randomJob"         , False                          , _SetHandlerRandomOption ) ,
                     "Random Scratch"       : ( "randomScratch"     , False                          , _SetHandlerRandomOption ) ,
                     "Scratch"              : ( "scratch"           , _QChemScratch                  , _SetHandlerPathOption   ) ,
                     "Number Of Threads"    : ( "numThreads"        , 1                              , None                    ) ,
                     "CDFT_Label"           : ( "CDFT"              , False                          , None                    ) ,  # KEVIN
                     "CDFTspin"             : ( "cdftSpinCons"      , None                           , None                    ) ,  # KEVIN
                     "CDFTcharge"           : ( "cdftChargeCons"    , None                           , None                    ) ,  # KEVIN
                     "CDFTspinCI1"          : ( "cdftSpinConsCI1"   , None                           , None                    ) ,  # KEVIN
                     "CDFTspinCI2"          : ( "cdftSpinConsCI2"   , None                           , None                    ) ,  # KEVIN
                     "CDFTchargeCI1"        : ( "cdftChargeConsCI1" , None                           , None                    ) ,  # KEVIN
                     "CDFTchargeCI2"        : ( "cdftChargeConsCI2" , None                           , None                    ) ,  # KEVIN
                     "Save Label"           : ( "save_label"        , None                           , _SetHandlerPathOption   ) ,  # KEVIN
                     "Original QCSCRATCH"   : ( "orig_qc_scratch"   , Path(os.environ[QCSCRATCH])    , None                    ) ,  # KEVIN
                     "GSM_Label"            : ( "GSM"               , False                          , None                    ) ,  # KEVIN
                     "Minimization_Label"   : ( "Minimization"      , False                          , None                    ) }  # KEVIN
    defaultKeys.update ( QCModel.defaultKeys )
    defaultLabel = "QChem"
    
    def __del__ ( self ):
        """Deallocation."""
        self.DeleteJobFiles ( )

    def _Initialize ( self ):
        super ( QCModelQChem, self )._Initialize ( )
        self.DeterminePaths ( )

    def AtomicCharges ( self, chargeModel = "Mulliken" ):
        """Atomic charges."""
        source = self._target.scratch.qchemOutputData 
        if   chargeModel == "Chelpg": return source.get ( "Chelpg Charges"   , None )
        elif chargeModel == "Lowdin": return source.get ( "Lowdin Charges"   , None )
        else:                         return source.get ( "Mulliken Charges" , None )

    def AtomicSpins ( self, chargeModel = "Mulliken" ):
        """Atomic spins."""
        source = self._target.scratch.qchemOutputData
        if   chargeModel == "Lowdin": return source.get ( "Lowdin Spins"     , None )
        else:                         return source.get ( "Mulliken Spins"   , None )

    def BondOrders ( self, chargeModel = None ):
        """Bond Orders - Mayer only."""
        return self._target.scratch.qchemOutputData.get ( "Mayer Bond Orders", None )

    def DeleteJobFiles ( self ):
        """Delete job files."""
        if self.deleteJobFiles:
            try:
                jobFiles = glob.glob ( os.path.join ( self.paths["Glob"] + ".*" ) )
                for jobFile in jobFiles: os.remove ( jobFile )
                scratch  = self.paths.get ( "Scratch", None )
                if scratch is not None: os.rmdir ( scratch )
                self.Pop ( "Paths" )
            except:
                pass

    def DeterminePaths ( self ):
        """Determine the paths needed by a QChem job."""
        paths = {}
        if self.randomJob: job = _RandomString ( )
        else:              job = self.job
        if self.randomScratch:
            scratch          = os.path.join ( self.scratch, _RandomString ( ) )
            paths["Scratch"] = scratch
        else:
            scratch          = self.scratch
        
        job_dir = Path ( scratch )
        if self.save_label is not None:
            job_dir /= self.save_label
            os.environ[QCSCRATCH] = str ( self.orig_qc_scratch / self.save_label )
        Path.mkdir ( job_dir, parents = True, exist_ok = True )
        
        # if not os.path.exists ( scratch ): os.mkdir ( scratch )
        # jobRoot       = os.path.join ( scratch, job )
        jobRoot       = job_dir / job
        paths["Glob"] = str(jobRoot)
        for ( key, ext ) in ( ( "Input"  , "inp"    ) ,
                              ( "Output" , "log"    ) ,
                              ( "PC"     , "pc"     ) ):
            paths[key] = f'{jobRoot}.{ext}'
            # paths[key] = "{:s}.{:s}".format ( jobRoot, ext )
        self.Set ( "Paths", paths, attribute = "paths" )

    def DipoleMoment ( self, target, center = None ):
        """Dipole Moment."""
        return target.scratch.qchemOutputData.get ( "Dipole", None )

    def Energy ( self ):
        """Calculate the quantum chemical energy."""
        target         = self._target
        doGradients    = target.scratch.doGradients
        qchemOutputData = {}
        target.scratch.qchemOutputData = qchemOutputData

        self.WriteInputFile ( doGradients, ( target.nbModel is not None ) )
        isOK = self.Execute ( )
        if not isOK: self.SaveErrorFiles ( "Error executing program." )
        isOK = self.ReadOutputFile ( target.scratch ) # . Returns whether converged or an error. 
        if not isOK: self.SaveErrorFiles ( "Error reading output file." )
        target.scratch.energyTerms["QChem QC"] = ( qchemOutputData["Energy"] * Units.Energy_Hartrees_To_Kilojoules_Per_Mole )

    def Execute ( self ):
        """Execute the job."""
        try:
            outFile = open ( self.paths["Output"], "w" )
            subprocess.check_call ( [ self.command, "-nt", str(self.numThreads), self.paths["Input"], self.paths["Output"], "save" ], stderr = outFile, stdout = outFile)
            outFile.close ( )
            return True
        except:
            return False

    # . Alpha/beta?
    def OrbitalEnergies ( self, target ):
        """Orbital energies and HOMO and LUMO indices."""
        return ( target.scratch.qchemOutputData.get ( "Orbital Energies", None ) ,
                 target.scratch.qchemOutputData.get ( "HOMO"            , -1   ) , 
                 target.scratch.qchemOutputData.get ( "LUMO"            , -1   ) ) 

    def ReadLowdinCharges(self, filename, scratch):
        # open qchem output file and extract line numbers of key lines in the file
        with open(filename) as qchem_file:
            qchem_file_lines = qchem_file.readlines()
            for index, line in enumerate(qchem_file_lines):
                if ('There are' in line) and ('shells and') in line:
                    num_shells = list(int(s) for s in line.split() if s.isdigit())[0]
                if 'Partial Lowdin       Populations for Occupied Alpha Orbitals' in line:
                    alpha_line_num = index
                if 'Partial Lowdin        Populations for Occupied Beta Orbitals' in line:
                    beta_line_num = index
                    break

        # compute line numbers bounding blocks of lowdin populations
        end_line_num = beta_line_num + (beta_line_num - alpha_line_num)
        header_lines = list(range(alpha_line_num+2, beta_line_num-2, num_shells+1))
        header_lines += list(range(beta_line_num+2, end_line_num-2, num_shells+1))
        ranges = [(header_lines[i]+1, header_lines[i]+num_shells+1) for i in range(len(header_lines))]

        # extract the lowdin population values from the fortran formatted table
        blocks = np.array([[line.split()[4:] for line in qchem_file_lines[block[0]:block[1]]] for block in ranges], dtype=float)
        populations = np.concatenate(blocks, axis=1)

        # map shells in the table to corresponding atoms
        atom_instances = [tuple(line.split()[1:3]) for line in qchem_file_lines[ranges[0][0]:ranges[0][1]]]

        # add the lowdin populations grouped by atom
        atom_populations = defaultdict(float)
        for index, atom_instance in enumerate(atom_instances):
            atom_populations[atom_instance] += sum(populations[index])

        # compute lowdin charges from populations
        atomic_numbers = {atom: PeriodicTable.AtomicNumber(atom[0]) for atom in atom_populations}
        lowdin_charges = Array.WithExtent(len(self.atomicNumbers))
        for index, (atom, population) in enumerate(atom_populations.items()):
            lowdin_charges[index] = atomic_numbers[atom] - population
        scratch["Lowdin Charges"] = lowdin_charges

    def ReadOutputFile ( self, target_scratch ):
        """Read an output file."""
        try:
            qchemOutputData = target_scratch.qchemOutputData
            n = len ( self.atomicNumbers )
            filename = self.paths["Output"]

            # cclib reads the qchem output file into a structured object
            data = cclib.io.ccread ( filename )
            
            # assuming the energy converged if cclib does not throw an exception
            scratch = { "Is Converged" : True }

            # Energy
            # Account for nuclear repulsion energy (ENUC) for QM/MM calculations  # KEVIN
            if target_scratch.doGradients:
                with open ( filename ) as fn:
                    for line in fn:
                        if "Etot" in line:
                            lsplit = line.split ()
                            ENUC = 2 * float ( lsplit[-2] )
                            scratch["Energy"] = ( ( data.scfenergies[-1] / Units.Energy_Hartrees_To_Electron_Volts ) - ENUC ) 
                        # Energy for QM only calculations
                        else:
                            scratch["Energy"] = ( data.scfenergies[-1] / Units.Energy_Hartrees_To_Electron_Volts )
            if self.cdftChargeConsCI1:
                with open ( filename ) as fn:
                    for line in fn:
                        if "E(Tot)" in line:
                            lsplit = line.split ()
                            E_Tot = lsplit[-1]
                            ENUC = 2 * float ( lsplit[-3] )
                            E_QM = float ( float ( E_Tot ) - float ( ENUC ) )
                            scratch["Energy"] = float ( E_QM )
            # Energy for QM only calculations
            else:
                 scratch["Energy"] = ( data.scfenergies[-1] / Units.Energy_Hartrees_To_Electron_Volts )

            # Get output data for CDFT-CI
            # Get CDFT-CI summary
            if self.cdftChargeConsCI1:
                CI_out = open ( "ResultsCI.txt", "w")
                with open ( filename ) as fn:
                    copy = False
                    for line in fn:
                        if line.startswith ( " CDFT-CI using one" ):
                            copy = True
                        elif line.startswith ( " ---" ):
                            copy = False
                        elif copy:
                            CI_out.write ( line )
                    CI_out.close ()
                # Energies 
                with open ( filename ) as fn:
                    for line in fn:
                        if "CDFT-CI Energy state 1" in line:
                            lsplit = line.split ()
                            E_1 = float ( lsplit[-1] )
                        elif "CDFT-CI Energy state 2" in line:
                            lsplit = line.split ()
                            E_2 = float ( lsplit[-1] )
                            scratch["Energy"] = float ( E_2 )
                            # Print table in log file
                            print ( "\n=============CDFT-CI Output=============" )
                            print ( "| State 1 Energy   = ", "{:>13.6f}".format ( E_1 ), "Eh |" )
                            print ( "| State 2 Energy   = ", "{:>13.6f}".format ( E_2 ), "Eh |" )
                            print ( "|                                      |" )
                # Electronic coupling constraint (H_ab)
                with open ( filename ) as fn:
                    for line in fn:
                        lines = fn.readlines ()
                        for i, line in enumerate ( lines ):
                            if line.startswith ( " CDFT-CI Hamiltonian matrix in orthogonalized" ):
                                new_line = lines[i+2]
                                H_ab = float ( new_line.split()[-1] )
                                print ( "| Coupling (H_ab)  = ", "{:>13.6f}".format ( H_ab ), "Eh |" )
                                print ( "========================================" )

            # collect mulliken charges
            if not self.cdftChargeConsCI1:
                # using pDynamo's built in array type
                mulliken = Array.WithExtent ( n )
                # cclib seems to fail to parse atomcharges with only one atom
                if n == 1: mulliken[0] = data.charge
                else:
                    for i in range ( n ):
                        mulliken[i] = data.atomcharges['mulliken'][i]
                scratch["Mulliken Charges"] = mulliken
            
            # Uncomment for Lowdin Charges  # KEVIN
            #self.ReadLowdinCharges(filename, scratch)
                
            # Get Gradients
            if target_scratch.doGradients:
                gradient = target_scratch.qcGradients3AU
                cclib_grad = data.grads[-1]
                for i in range ( cclib_grad.shape[0] ):
                    for j in range ( cclib_grad.shape[1] ):
                        gradient[i][j] = cclib_grad[i][j]

            # update the object's output data variable with the new output file
            qchemOutputData.update ( scratch ) 
            return True
        except Exception as e:
            print('Exception in QCModelQChem.ReadOutputFile: ', e)
            return False

    def SaveErrorFiles ( self, message ):
        """Save the input and output files for inspection if there is an error."""
        for key in ( "Input" , "Output" , "PC" ):
            path = self.paths[key]
            if os.path.exists ( path ):
                ( head, tail ) = os.path.split ( path )
                os.rename ( path, os.path.join ( head, _DefaultErrorPrefix + tail ) )
        ( head, tail ) = os.path.split ( self.paths["Glob"] )
        raise QCModelError ( message + "\nCheck the files \"{:s}*\".".format ( os.path.join ( head, _DefaultErrorPrefix + tail ) ) )

    def SummaryEntries ( self ):
        """Summary entries."""
        entries = super ( QCModelQChem, self ).SummaryEntries ( )
        if self.keywords is not None:
            n = len ( self.keywords )
            entries.append ( ( "Keywords", "{:s}".format ( "/".join ( self.keywords[0:min(2,n)] ) ) ) )
        for key in ( "Delete Job Files", "Random Job Name", "Random Scratch" ):
            entries.append ( ( key, "{:s}".format ( repr ( self._entries[key] ) ) ) )
        return entries

    # . Coordinates are written in angstroms
    def WriteInputFile ( self, doGradients, doQCMM):
        """Write an input file."""
        with open ( self.paths["Input"], "w" ) as inFile:
            
            inFile.write ( "#\n" )
            inFile.write ( "# QChem Job\n" )
            inFile.write ( "#\n\n" )
            
            # Write $rem section  # KEVIN
            inFile.write ( "$rem\n" )
            if doGradients: mode = "JOBTYPE              FORCE"
            else          : mode = "JOBTYPE              SP"
            inFile.write ( " " + mode)
        
            # Read in MOs if they exist  # KEVIN
            if self.GSM or self.Minimization or self.CDFT:
                if Path ( f'{os.environ["QCSCRATCH"]}/save/53.0' ).exists():
                    inFile.write ( "\n SCF_GUESS            READ\n" )

            # Manually defined keywords  # KEVIN
            inFile.write ( " ".join ( self.keywords ) )

            # Write Löwdin Populations  # KEVIN
            #inFile.write ( "\n LOWDIN_POPULATION    TRUE\n" )

            # Determine if to do CDFT for internal GSM nodes # KEVIN
            saved_orbitals_exist = Path ( f'{os.environ["QCSCRATCH"]}/save/53.0' ).exists()
            cdft_constraints_exist = any ( ( self.cdftSpinCons, self.cdftChargeCons, self.cdftChargeConsCI1 ) )
            do_cdft = cdft_constraints_exist and saved_orbitals_exist

            # Definition for writing CDFT keyword in $rem section  # KEVIN
            def write_cdft_rem ():
                inFile.write ( "\n CDFT                 TRUE\n" )

            # Write CDFT-CI $rem keyword
            def write_cdftci_rem ():
                inFile.write ( "\n CDFTCI               TRUE\n" )

            # Write CDFT keywords for internal GSM nodes, Minimization, or SP
            if do_cdft or self.CDFT:
                write_cdft_rem ()

            # Write CDFT keywords for end nodes in GSM
            elif self.GSM and self.orig_qc_scratch and do_cdft:
                write_cdft_rem ()

            # Write CDFT-CI $rem keyword
            if self.cdftChargeConsCI1:
                write_cdftci_rem ()

            # Complete the $rem section  # KEVIN
            inFile.write ( "$end\n\n" )
            
            def write_cdft_constraint(atom_range, range_coefficient, constraint_type):
                first_atom_index = list ( self.pureQCAtoms ).index ( atom_range.start ) + 1
                last_atom_index = first_atom_index + len ( list ( atom_range ) ) - 1
                inFile.write ( f'{range_coefficient:5d} {first_atom_index:4d} {last_atom_index:4d} {constraint_type}\n' )
                # Loop through link atoms, adding any that have a partner in the constraint region to the constraint
                for i, ( boundary_atom_index, qc_atom_index, distance ) in enumerate ( self.baQCPartners ):
                    if qc_atom_index in atom_range:
                        link_atom_index = len ( self.pureQCAtoms ) + i + 1
                        inFile.write( f'{range_coefficient:5d} {link_atom_index:4d} {link_atom_index:4d} {constraint_type}\n' )

            def write_cdft_constraint_section(constraints, constraint_type):
                for constraint_value, atom_ranges in constraints:
                    inFile.write( "  " + str ( constraint_value ) + "\n" )
                    # Loop through atom ranges included within the current cdft constraint
                    for atom_range, range_coefficient in atom_ranges.items ():
                        write_cdft_constraint(atom_range, range_coefficient, constraint_type)

            # Definition for writing CDFT or CDFT-CI $cdft sections
            def write_cdft_section():
                inFile.write ( "$cdft\n" )
                # Normal CDFT
                if self.cdftSpinCons:
                    write_cdft_constraint_section ( self.cdftSpinCons, 's' )
                if self.cdftChargeCons:
                    write_cdft_constraint_section ( self.cdftChargeCons, '' )
                # CDFT-CI
                if self.cdftChargeConsCI1:
                    write_cdft_constraint_section ( self.cdftChargeConsCI1, '' )
                    write_cdft_constraint_section ( self.cdftSpinConsCI1, 's' )
                    inFile.write ( " -----------------\n" )
                    write_cdft_constraint_section ( self.cdftChargeConsCI2, '' )
                    write_cdft_constraint_section ( self.cdftSpinConsCI2, 's' )
                inFile.write ( "$end\n\n" )
            
            # Write CDFT section for internal GSM nodes, Minimization, or SP
            if do_cdft or self.CDFT:
                write_cdft_section ()

            # Write CDFT section for end nodes in GSM
            elif do_cdft and self.GSM and self.orig_qc_scratch:
                write_cdft_section ()

            # qm_atoms section
            if doQCMM:
                inFile.write ( "$qm_atoms\n" )
                inFile.write ( f' 1:{ len ( self.atomicNumbers ) }\n' )
                inFile.write ( "$end\n\n" )

            # force_field_params section
            if doQCMM:
                inFile.write ( "$force_field_params\n" )
                inFile.write ( " NumAtomTypes 999\n" )

                pc_file = np.loadtxt ( self.paths["PC"], skiprows = 1, usecols = [ 1, 2, 3, 4 ] )
                pc_elements = np.loadtxt ( self.paths["PC"], skiprows = 1, usecols = [0], dtype= str )
                pc_charges = pc_file[:,0]
                pc_coords = pc_file[:,1:]
                
                # create a map from each unique value of atom charge to a uniquely numbered atom type
                unique_charges = np.unique(pc_charges)
                atom_types = np.arange ( -101, -102 - len ( unique_charges ), step = -1 )
                atom_type_charges = np.array([0.0] + list(unique_charges))
                charges_to_atom_types = {atom_type_charges[i]: atom_types[i] for i in range(1,len(atom_types))}

                # Write AtomType information
                
                label_col = np.full_like ( atom_types, ' AtomType', dtype=np.object )
                data = np.column_stack ( ( label_col, atom_types, atom_type_charges,
                                           np.ones_like ( atom_types ), np.zeros_like ( atom_types ) ) )
                np.savetxt ( inFile, data, fmt='%s' ) # appends the atom type data into input file
                inFile.write ( "$end\n\n" )

            # $molecule section
            qcCoordinates3 = self._target.scratch.qcCoordinates3
            inFile.write ( "$molecule\n" )
            inFile.write ( f"{self.electronicState.charge} {self.electronicState.multiplicity}\n" )

            # Add QM coordinates
            atom_type = -101
            for (i, n) in enumerate(self.atomicNumbers):
                inFile.write(f'{PeriodicTable.Symbol(n):<4s}')
                for j in range(3):
                   inFile.write(f'{qcCoordinates3[i,j]:20.10f}')
                if doQCMM:
                    inFile.write(f' {atom_type:0d} 0 0 0 0')
                inFile.write('\n')

            # Add MM coordinates
            if doQCMM:
                for i in range(pc_coords.shape[0]):
                    inFile.write(f'{pc_elements[i]:<4s}')
                    for j in range(3):
                        inFile.write(f'{pc_coords[i,j]:20.10f}')
                    atom_type = charges_to_atom_types[pc_charges[i]]
                    inFile.write(f' {atom_type:0d} 0 0 0 0\n')

            inFile.write ( "$end\n\n" )

#===================================================================================================================================
# . Testing.
#===================================================================================================================================
if __name__ == "__main__" :
    pass

