import sys, os

import copy

import logging

import numpy as np

from VLMP.components.models import modelBase

from pyGrained.utils.data import getData
import re

class Calvados(modelBase):
    """
    Component name: Calvados
    Component type: model

    Author: Salvatore Assenza
    Date: 14/03/2024

    Model for multidomain and intrinsic disordered proteins.

    Reference: http://dx.doi.org/10.1002/pro.5172

    """

    def __init__(self,name,**params):
        super().__init__(_type = self.__class__.__name__,
                         _name= name,
                         availableParameters = {"sequence", "coordinates_file", "bonds_file", "domains_file", "interfaces_file_A", "interfaces_file_B", "ionicStrength", "pH", "version", "thresholdInterface", "thresholdDomain"},
                         requiredParameters  = {"sequence"},
                         definedSelections   = {"particleId"},
                         **params)

        CALVADOS1 = {
            'ALA': {"lambda":0.0011162643859539, "sigma":5.04},  # Alanine
            'ARG': {"lambda":0.7249915947715210, "sigma":6.559999999999999},  # Arginine
            'ASN': {"lambda":0.4383272997027280, "sigma":5.68},  # Asparagine
            'ASP': {"lambda":0.0291821237763497, "sigma":5.579999999999999},  # Aspartic Acid
            'CYS': {"lambda":0.610362354303913, "sigma":5.479999999999999},  # Cysteine
            'GLN': {"lambda":0.3268188050525210, "sigma":6.02},  # Glutamine
            'GLU': {"lambda":0.0061002816086497, "sigma":5.920000000000001},  # Glutamic Acid
            'GLY': {"lambda":0.7012713677972460, "sigma":4.5},  # Glycine
            'HIS': {"lambda":0.4651948082346980, "sigma":6.08},  # Histidine
            'ILE': {"lambda":0.6075268330845270, "sigma":6.18},  # Isoleucine
            'LEU': {"lambda":0.5563020305733200, "sigma":6.18},  # Leucine
            'LYS': {"lambda":0.0586171731586979, "sigma":6.36},  # Lysine
            'MET': {"lambda":0.7458993420826710, "sigma":6.18},  # Methionine
            'PHE': {"lambda":0.9216959832175940, "sigma":6.36},  # Phenylalanine
            'PRO': {"lambda":0.3729641853599350, "sigma":5.559999999999999},  # Proline
            'SER': {"lambda":0.4648570130065610, "sigma":5.18},  # Serine
            'THR': {"lambda":0.5379777613307020, "sigma":5.62},  # Threonine
            'TRP': {"lambda":0.9844235478393930, "sigma":6.779999999999999},  # Tryptophan
            'TYR': {"lambda":0.9950108229594320, "sigma":6.459999999999999},  # Tyrosine
            'VAL': {"lambda":0.4185006852559870, "sigma":5.860000000000001}   # Valine
        }
        sigma_loc = np.mean([CALVADOS1[x]['sigma'] for x in CALVADOS1])
        CALVADOS1['ZZZ'] = {}
        CALVADOS1['ZZZ']['lambda'] = 0.0
        CALVADOS1['ZZZ']['sigma'] = sigma_loc

        CALVADOS2 = {
            'ALA': {"lambda":0.2743297969040348, "sigma":5.04},  # Alanine
            'ARG': {"lambda":0.7307624767517166, "sigma":6.559999999999999},  # Arginine
            'ASN': {"lambda":0.4255859009787713, "sigma":5.68},  # Asparagine
            'ASP': {"lambda":0.0416040480605567, "sigma":5.579999999999999},  # Aspartic Acid
            'CYS': {"lambda":0.5615435099141777, "sigma":5.479999999999999},  # Cysteine
            'GLN': {"lambda":0.3934318551056041, "sigma":6.02},  # Glutamine
            'GLU': {"lambda":0.0006935460962935, "sigma":5.920000000000001},  # Glutamic Acid
            'GLY': {"lambda":0.7058843733666401, "sigma":4.5},  # Glycine
            'HIS': {"lambda":0.4663667290557992, "sigma":6.08},  # Histidine
            'ILE': {"lambda":0.5423623610671892, "sigma":6.18},  # Isoleucine
            'LEU': {"lambda":0.6440005007782226, "sigma":6.18},  # Leucine
            'LYS': {"lambda":0.1790211738990582, "sigma":6.36},  # Lysine
            'MET': {"lambda":0.5308481134337497, "sigma":6.18},  # Methionine
            'PHE': {"lambda":0.8672358982062975, "sigma":6.36},  # Phenylalanine
            'PRO': {"lambda":0.3593126576364644, "sigma":5.559999999999999},  # Proline
            'SER': {"lambda":0.4625416811611541, "sigma":5.18},  # Serine
            'THR': {"lambda":0.3713162976273964, "sigma":5.62},  # Threonine
            'TRP': {"lambda":0.9893764740371644, "sigma":6.779999999999999},  # Tryptophan
            'TYR': {"lambda":0.9774611449343455, "sigma":6.459999999999999},  # Tyrosine
            'VAL': {"lambda":0.2083769608174481, "sigma":5.860000000000001}   # Valine
        }
        sigma_loc = np.mean([CALVADOS2[x]['sigma'] for x in CALVADOS2])
        CALVADOS2['ZZZ'] = {}
        CALVADOS2['ZZZ']['lambda'] = 0.0
        CALVADOS2['ZZZ']['sigma'] = sigma_loc

        CALVADOS3 = {
            'ALA': {"lambda":0.3377244362031627, "sigma":5.04},  # Alanine
            'ARG': {"lambda":0.7407902764839954, "sigma":6.559999999999999},  # Arginine
            'ASN': {"lambda":0.3706962163690402, "sigma":5.68},  # Asparagine
            'ASP': {"lambda":0.092587557536158, "sigma":5.579999999999999},  # Aspartic Acid
            'CYS': {"lambda":0.5922529084601322, "sigma":5.479999999999999},  # Cysteine
            'GLN': {"lambda":0.3143449791669133, "sigma":6.02},  # Glutamine
            'GLU': {"lambda":0.000249590539426, "sigma":5.920000000000001},  # Glutamic Acid
            'GLY': {"lambda":0.7538308115197386, "sigma":4.5},  # Glycine
            'HIS': {"lambda":0.4087176216525476, "sigma":6.08},  # Histidine
            'ILE': {"lambda":0.5130398874425708, "sigma":6.18},  # Isoleucine
            'LEU': {"lambda":0.5548615312993875, "sigma":6.18},  # Leucine
            'LYS': {"lambda":0.1380602542039267, "sigma":6.36},  # Lysine
            'MET': {"lambda":0.5170874160398543, "sigma":6.18},  # Methionine
            'PHE': {"lambda":0.8906449355499866, "sigma":6.36},  # Phenylalanine
            'PRO': {"lambda":0.3469777523519372, "sigma":5.559999999999999},  # Proline
            'SER': {"lambda":0.4473142572693176, "sigma":5.18},  # Serine
            'THR': {"lambda":0.2672387936544146, "sigma":5.62},  # Threonine
            'TRP': {"lambda":1.033450123574512, "sigma":6.779999999999999},  # Tryptophan
            'TYR': {"lambda":0.950628687301107, "sigma":6.459999999999999},  # Tyrosine
            'VAL': {"lambda":0.2936174211771383, "sigma":5.860000000000001}   # Valine
        }
        sigma_loc = np.mean([CALVADOS3[x]['sigma'] for x in CALVADOS3])
        CALVADOS3['ZZZ'] = {}
        CALVADOS3['ZZZ']['lambda'] = 0.0
        CALVADOS3['ZZZ']['sigma'] = sigma_loc

        CALVADOS = {
            1: CALVADOS1,
            2: CALVADOS2,
            3: CALVADOS3
        }

        ############################################################

        sequence = params["sequence"]
        ionicStrength = params.get("ionicStrength", 150.0)
        pH = params.get("pH", 7.4)
        version = params.get("version", 3)
        coordinates_file = params.get("coordinates_file", None)
        bonds_file = params.get("bonds_file", None)
        domains_file = params.get("domains_file", None)
        interfaces_file_A = params.get("interfaces_file_A", None)
        interfaces_file_B = params.get("interfaces_file_B", None)
        thresholdInterface = params.get("thresholdInterface", 9.0)
        thresholdDomain = params.get("thresholdDomain", 9.0)
        
        Tloc = self.getEnsemble().getEnsembleComponent("temperature")
        dielectricConstant = 5321./Tloc + 233.76 - 0.9297*Tloc + 0.1417*1e-2*Tloc*Tloc - 0.8292*1e-6*Tloc*Tloc*Tloc
        kBT = self.getUnits().getConstant("KBOLTZ")*Tloc
        bjerrumLength = 1.6e-19*1.6e-19/(4*np.pi*8.859e-12*dielectricConstant*kBT)/4186*6.022e23
        debyeLength= 1./np.sqrt(8*np.pi*bjerrumLength*6.022e23*ionicStrength)*1e10

        # Bonded energy parameters
        Kb = 8033./418.6 ### converted from kJ/mol nm^2 to Kcal/mol Ang^2
        KENM = 700./418.6 ### converted from kJ/mol nm^2 to Kcal/mol Ang^2  (not sure about a factor of 2)
        r0 = 3.8 ### 0.38 nm
        self.logger.info(f"Kb = {Kb}, r0 = {r0}")

        # Non-bonded energy parameters
        epsilon = 0.2 ### Kcal/mol
        #cutOffLJ = 20.0 ### 2.0 nm
        cutOffLJ = 22.0 ### 2.0 nm
        cutOffDH = 40.0 ### 4 nm
        skin = 4.0 ### 0.4 nm
        cutOffVerletFactor = max(cutOffLJ, cutOffDH)
        cutOffVerletFactor = (cutOffVerletFactor + skin)/cutOffVerletFactor
        self.logger.info(f"Dielectric = {dielectricConstant}, epsilon = {epsilon}, cutOffLJ = {cutOffLJ}, cutOffDH = {cutOffDH}")
        
        ############################################################
        ######################  Import files  ######################
        ############################################################
        if domains_file != None:
            if coordinates_file == None:
                self.logger.error(f"[Calvados] If domains are provided, a coordinates file must be also present")
                raise Exception("[Calvados] Coordinates file not provided")

        if interfaces_file_A != None or interfaces_file_B != None:
            if coordinates_file == None:
                self.logger.error(f"[Calvados] If interfaces are provided, a coordinates file must be also present")
                raise Exception("[Calvados] Coordinates file not provided")

        if interfaces_file_A != None and interfaces_file_B == None:
            self.logger.error(f"[Calvados] Either both interfaces files are provided, or none of them")
            raise Exception("[Calvados] Interface file not provided")

        if interfaces_file_A == None and interfaces_file_B != None:
            self.logger.error(f"[Calvados] Either both interfaces files are provided, or none of them")
            raise Exception("[Calvados] Interface file not provided")

        if coordinates_file != None:
            if not os.path.exists(coordinates_file):
                self.logger.error(f"[Calvados] Coordinates file not found (selected: {coordinates_file})")
                raise Exception("[Calvados] File not found")
        if bonds_file != None:
            if not os.path.exists(bonds_file):
                self.logger.error(f"[Calvados] Bonds file not found (selected: {bonds_file})")
                raise Exception("[Calvados] File not found")
        if domains_file != None:
            if not os.path.exists(domains_file):
                self.logger.error(f"[Calvados] Domains file not found (selected: {domains_file})")
                raise Exception("[Calvados] File not found")

        if coordinates_file != None:
            coordinates = np.loadtxt(coordinates_file)
        else:
            coordinates = []
            xloc = 0
            yloc = 0
            floc = 1
            for i in range(len(sequence)):
                coordinates.append([xloc, yloc, 0])
                if floc == 1:
                    xloc += r0
                else:
                    yloc += r0
                floc = 1 - floc

        if len(coordinates) != len(sequence):
            self.logger.error("[Calvados] Number of coordinates ({:d}) does not match sequence length ({:d})".format(len(coordinates), len(sequence)))
            raise Exception("[Calvados] Mismatch between coordinates file and sequence")
        
        ### import bonds
        bonds = []
        if bonds_file != None:
            with open(bonds_file, 'r') as file:
                lines = file.readlines()
                for iline in range(len(lines)):
                    line = lines[iline]
                    line_form = line.rsplit()
                    if len(line_form) == 1: ### range expected
                        field = line_form[0]
                        is_range = re.search(r'\d+-\d+', field)
                        if is_range:  ## detect presence of range
                            i1 = int(field.rsplit('-')[0])
                            i2 = int(field.rsplit('-')[1])
                            for i in range(i1, i2):
                                bonds.append([i, i+1])
                        else:
                            self.logger.error("[Calvados] Bonds file {:s}, error on line {:d}: with one field is present it is assumed a range to be present, but '-' was not detected".format(bonds_file, iline+1))
                            raise Exception("[Calvados] Wrong format in bonds file")
                    elif len(line_form) == 2: ### single bond expected
                        if re.search(r'\d+-\d+', line_form[0]) or re.search(r'\d+-\d+', line_form[1]):
                            self.logger.error("[Calvados] Bonds file {:s}, error on line {:d}: with two fields present it is assumed a single bond to be listed, but a range was instead detected".format(bonds_file, iline+1))
                            raise Exception("[Calvados] Wrong format in bonds file")
                        bonds.append([int(line_form[0]), int(line_form[1])])
                    else:
                        self.logger.error("[Calvados] Bonds file {:s}, error on line {:d}: invalid input. Only one field is allowed for a range (e.g. 4-7) or two fields for a single bond (e.g. 4 7)".format(bonds_file, iline+1))
                        raise Exception("[Calvados] Wrong format in bonds file")
        else:
            for i in range(0,len(sequence)-1):
                bonds.append([i, i+1])

        ### import domains
        domains = []
        if domains_file != None:
            with open(domains_file, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    line_form = line.rsplit()
                    domain_local = []
                    for field in line_form:
                        is_range = re.search(r'\d+-\d+', field)
                        if is_range:  ## detect presence of range
                            i1 = int(field.rsplit('-')[0])
                            i2 = int(field.rsplit('-')[1])
                            for i in range(i1, i2+1):
                                domain_local.append(i)
                        else:
                            domain_local.append(int(field))
                    domains.append(domain_local)

        ### import interfaces
        interfaces_A = []
        if interfaces_file_A != None:
            with open(interfaces_file_A, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    line_form = line.rsplit()
                    interface_local = []
                    for field in line_form:
                        is_range = re.search(r'\d+-\d+', field)
                        if is_range:  ## detect presence of range
                            i1 = int(field.rsplit('-')[0])
                            i2 = int(field.rsplit('-')[1])
                            for i in range(i1, i2+1):
                                interface_local.append(i)
                        else:
                            interface_local.append(int(field))
                    interfaces_A.append(interface_local)

        interfaces_B = []
        if interfaces_file_B != None:
            with open(interfaces_file_B, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    line_form = line.rsplit()
                    interface_local = []
                    for field in line_form:
                        is_range = re.search(r'\d+-\d+', field)
                        if is_range:  ## detect presence of range
                            i1 = int(field.rsplit('-')[0])
                            i2 = int(field.rsplit('-')[1])
                            for i in range(i1, i2+1):
                                interface_local.append(i)
                        else:
                            interface_local.append(int(field))
                    interfaces_B.append(interface_local)

        if len(interfaces_A) != len(interfaces_B):
            self.logger.error("[Calvados] Interfaces files must have the same length")
            raise Exception("[Calvados] Incompatible interfaces files")

        ############################################################
        ######################  Set up model  ######################
        ############################################################

        units = self.getUnits()

        if units.getName() != "KcalMol_A":
            self.logger.error(f"[Calvados3] Units are not set correctly. Please set units to \"KcalMol_A\" (selected: {units.getName()})")
            raise Exception("Not correct units")

        ############################################################
        # Set up types

        names   = getData("aminoacids")
        masses  = getData("aminoacidMasses")
        radii   = getData("aminoacidRadii")
        charges = getData("aminoacidCharges")

        # In this model the charge of HIS is dictated by the Henderson-Hasselbach equation with pKa=6
        charges["HIS"] = 1./(1+np.power(10, pH-6.0))

        # CHANGE RADII FOR SURFACE POTENTIAL
        for x in radii:
            radii[x] = 3.0

        #Invert names dict
        aminoacids_3to1 = {v: k for k, v in names.items()}

        ### Add amino acid Z: ZZZ, corresponding to average mass, radius and sigma, zero charge and zero hydrophobicity
        aminoacids_3to1['Z'] = 'ZZZ'
        names['ZZZ'] = 'Z'
        charges['ZZZ'] = 0
        masses['ZZZ'] = np.mean([masses[x] for x in masses])
        radii['ZZZ'] = np.mean([radii[x] for x in radii])

        ############################################################

        types = self.getTypes()
        for nm in names:
            types.addType(name = nm,
                          mass = masses[nm],
                          radius = radii[nm],
                          charge = charges[nm])

        ############################################################

        # State

        state = {}
        state["labels"] = ["id","position"]
        state["data"]   = []

        for i,s in enumerate(sequence):
            pos = np.array(coordinates[i])
            pos = pos.tolist()
            state["data"].append([i,pos])

        # Structure

        structure = {}
        structure["labels"] = ["id","type", "modelId"]
        structure["data"]   = []

        for i,s in enumerate(sequence):
            try:
                structure["data"].append([i,aminoacids_3to1[s], 0])
            except:
                self.logger.error(f"[Calvados3] Aminoacid {s} not recognized")
                raise Exception("Aminoacid not recognized")

        # Forcefield

        forcefield = {}
        exclusions = {}
        for i in range(len(sequence)):
            exclusions[i] = []

        # chain bonds 
        forcefield["bonds"] = {}
        forcefield["bonds"]["type"]             = ["Bond2","HarmonicCommon_K_r0"]
        forcefield["bonds"]["parameters"]       = {}
        forcefield["bonds"]["parameters"]["K"]  = Kb
        forcefield["bonds"]["parameters"]["r0"] = r0
        forcefield["bonds"]["labels"]           = ["id_i","id_j"]
        forcefield["bonds"]["data"]             = []

        for ibnd in range(len(bonds)):
            bnd = bonds[ibnd]
            i = int(bnd[0])
            j = int(bnd[1])
            forcefield["bonds"]["data"].append([i,j])
            exclusions.setdefault(i,[]).append(j)
            exclusions.setdefault(j,[]).append(i)

        # ENM bonds
        if len(domains)>0:
            forcefield["ENM"] = {}
            forcefield["ENM"]["type"]             = ["Bond2","Harmonic"]
            forcefield["ENM"]["parameters"]       = {}
            forcefield["ENM"]["labels"]           = ["id_i", "id_j", "K", "r0"]
            forcefield["ENM"]["data"]             = []
            
            for domain_loc in domains:
                for i_d in range(0,len(domain_loc)):
                    for j_d in range(i_d+1,len(domain_loc)):
                        i = domain_loc[i_d]
                        j = domain_loc[j_d]
                        pos_i = np.array(coordinates[i])
                        pos_j = np.array(coordinates[j])
                        distance = float(np.linalg.norm(pos_i-pos_j))
                        if distance <= thresholdDomain:
                            forcefield["ENM"]["data"].append([i, j, KENM, distance])
                            exclusions.setdefault(i,[]).append(j)
                            exclusions.setdefault(j,[]).append(i)

        # Interfaces
        if len(interfaces_A)>0:
            forcefield["Interfaces"] = {}
            forcefield["Interfaces"]["type"]             = ["Bond2","Harmonic"]
            forcefield["Interfaces"]["parameters"]       = {}
            forcefield["Interfaces"]["labels"]           = ["id_i", "id_j", "K", "r0"]
            forcefield["Interfaces"]["data"]             = []
    
            for i_interface in range(len(interfaces_A)):
                interface_loc_A = interfaces_A[i_interface]
                interface_loc_B = interfaces_B[i_interface]
                for i_d in range(len(interface_loc_A)):
                    for j_d in range(len(interface_loc_B)):
                        i = interface_loc_A[i_d]
                        j = interface_loc_B[j_d]
                        pos_i = np.array(coordinates[i])
                        pos_j = np.array(coordinates[j])
                        distance = float(np.linalg.norm(pos_i-pos_j))
                        if distance <= thresholdInterface:
                            forcefield["Interfaces"]["data"].append([i, j, KENM, distance])
                            exclusions.setdefault(i,[]).append(j)
                            exclusions.setdefault(j,[]).append(i)

        #NL
        for exc in exclusions:
            exclusions[exc] = sorted(list(set(exclusions[exc])))
        forcefield["nl"] = {}
        forcefield["nl"]["type"]       = ["VerletConditionalListSet", "nonExcluded"]
        forcefield["nl"]["parameters"] = {"cutOffVerletFactor": cutOffVerletFactor}
        forcefield["nl"]["labels"]     = ["id", "id_list"]
        forcefield["nl"]["data"] = []

        for i in range(len(sequence)):
            forcefield["nl"]["data"].append([i,exclusions[i]])

        # HYDROPHOBIC
        sigma_max = np.max([CALVADOS[version][x]["sigma"] for x in CALVADOS[version]])
        forcefield["hydrophobic"] = {}
        forcefield["hydrophobic"]["type"]       = ["NonBonded", "SplitLennardJones"]
        forcefield["hydrophobic"]["parameters"] = {"cutOffFactor": cutOffLJ/sigma_max,
                                                   "epsilon_r": epsilon,
                                                   "epsilon_a": epsilon,
                                                   "condition":"nonExcluded"}
        forcefield["hydrophobic"]["labels"] = ["name_i", "name_j", "epsilon", "sigma"]
        forcefield["hydrophobic"]["data"]   = []

        for t1 in names:
            for t2 in names:
                eps = 0.5*(CALVADOS[version][t1]["lambda"] + CALVADOS[version][t2]["lambda"])
                sigma = 0.5*(CALVADOS[version][t1]["sigma"] + CALVADOS[version][t2]["sigma"])
                forcefield["hydrophobic"]["data"].append([t1,t2,eps,sigma])

        #DH
        forcefield["DH"] = {}
        forcefield["DH"]["type"]       = ["NonBonded", "DebyeHuckel"]
        forcefield["DH"]["parameters"] = {"cutOffFactor": cutOffDH/debyeLength,
                                          "debyeLength": debyeLength,
                                          "dielectricConstant": dielectricConstant,
                                          "condition":"nonExcluded"}

        ############################################################

        self.setState(state)
        self.setStructure(structure)
        self.setForceField(forcefield)

    def processSelection(self,**params):

        sel = []

        if "particleId" in params:
            sel += params["particleId"]
        #if "domain" in params:
        #    selectedDomains = params["domain"]

        return sel
