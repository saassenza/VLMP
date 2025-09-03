import sys, os

import copy

import logging

import numpy as np

from VLMP.components.models import modelBase

from pyGrained.utils.data import getData
import re

class IDP_FRET(modelBase):
    """
    Component name: IDP_FRET
    Component type: model

    Author: Salvatore Assenza
    Date: 23/12/2024

    Model for multidomain and intrinsic disordered proteins.

    Reference: https://doi.org/10.1103/PhysRevE.90.042709

    """

    def __init__(self,name,**params):
        super().__init__(_type = self.__class__.__name__,
                         _name= name,
                         availableParameters = {"sequence", "coordinates_file", "bonds_file", "angles_file", "dihedrals_file", "domains_file", "interfaces_file_A", "interfaces_file_B", "ionicStrength", "pH", "hydrophobicityScale", "lambdaSoft", "lambdaGroupsFile", "alphaSoft", "nSoft"},
                         requiredParameters  = {"sequence"},
                         definedSelections   = {"particleId"},
                         **params)

        Monera = {
            'ALA': {"lambda":0.62, "sigma":4.80},  # Alanine
            'ARG': {"lambda":0.26, "sigma":4.80},  # Arginine
            'ASN': {"lambda":0.17, "sigma":4.80},  # Asparagine
            'ASP': {"lambda":0, "sigma":4.80},  # Aspartic Acid
            'CYS': {"lambda":0.67, "sigma":4.80},  # Cysteine
            'GLN': {"lambda":0.28, "sigma":4.80},  # Glutamine
            'GLU': {"lambda":0.15, "sigma":4.80},  # Glutamic Acid
            'GLY': {"lambda":0.35, "sigma":4.80},  # Glycine
            'HIS': {"lambda":0.40, "sigma":4.80},  # Histidine
            'ILE': {"lambda":0.99, "sigma":4.80},  # Isoleucine
            'LEU': {"lambda":0.98, "sigma":4.80},  # Leucine
            'LYS': {"lambda":0.20, "sigma":4.80},  # Lysine
            'MET': {"lambda":0.83, "sigma":4.80},  # Methionine
            'PHE': {"lambda":1.00, "sigma":4.80},  # Phenylalanine
            'PRO': {"lambda":0.05, "sigma":4.80},  # Proline
            'SER': {"lambda":0.32, "sigma":4.80},  # Serine
            'THR': {"lambda":0.43, "sigma":4.80},  # Threonine
            'TRP': {"lambda":0.97, "sigma":4.80},  # Tryptophan
            'TYR': {"lambda":0.75, "sigma":4.80},  # Tyrosine
            'VAL': {"lambda":0.84, "sigma":4.80}   # Valine
        }
        sigma_loc = np.mean([Monera[x]['sigma'] for x in Monera])
        lambda_loc = np.mean([Monera[x]['lambda'] for x in Monera])
        Monera['NOH'] = {}
        Monera['NOH']['lambda'] = 0.0
        Monera['NOH']['sigma'] = sigma_loc
        Monera['AVE'] = {}
        Monera['AVE']['lambda'] = lambda_loc
        Monera['AVE']['sigma'] = sigma_loc

        hydrophobicity = {
            'Monera': Monera
        }

        ############################################################

        sequence = params["sequence"]
        ionicStrength = params.get("ionicStrength", 150.0)
        pH = params.get("pH", 7.4)
        hydrophobicityScale = params.get("hydrophobicityScale", 'Monera')
        coordinates_file = params.get("coordinates_file", None)
        bonds_file = params.get("bonds_file", None)
        angles_file = params.get("angles_file", None)
        dihedrals_file = params.get("dihedrals_file", None)
        domains_file = params.get("domains_file", None)
        interfaces_file_A = params.get("interfaces_file_A", None)
        interfaces_file_B = params.get("interfaces_file_B", None)
        lambdaSoft = params.get("lambdaSoft", 1.0)
        alphaSoft = params.get("alphaSoft", 0.5)
        nSoft = params.get("nSoft", 2)
        lambdaGroupsFile = params.get("lambdaGroupsFile", None)
        
        Tloc = self.getEnsemble().getEnsembleComponent("temperature")
        dielectricConstant = 5321./Tloc + 233.76 - 0.9297*Tloc + 0.1417*1e-2*Tloc*Tloc - 0.8292*1e-6*Tloc*Tloc*Tloc
        kBT = self.getUnits().getConstant("KBOLTZ")*Tloc
        bjerrumLength = 1.6e-19*1.6e-19/(4*np.pi*8.859e-12*dielectricConstant*kBT)/4186*6.022e23
        debyeLength= 1./np.sqrt(8*np.pi*bjerrumLength*6.022e23*ionicStrength)*1e10

        # Bonded energy parameters (apart from dihedrals)
        Kb = 274.899 ### Kcal/mol Ang^2 obtained from kBT/0.046^2 at T=293 K
        r0 = 3.9
        self.logger.info(f"Kb = {Kb}, r0 = {r0}")
        Ka = 8.60482 ### Kcal/mol obtained from kBT/0.26^2 at T=293 K
        theta0 = 2.12 ### rad
        KENM = 700./418.6 ### converted from kJ/mol nm^2 to Kcal/mol Ang^2  (not sure about a factor of 2)
        
        # Bonded energy parameters (dihedrals, to be converted to kcal/mol and formula of UAMMD)
        dihedrals_values_article_sin = {1:-0.175, 2:-0.093, 3:0.030, 4:0.030}  ### units of kBT
        dihedrals_values_article_cos = {1:0.705, 2:-0.313, 3:-0.079, 4:0.041}  ### units of kBT
        K_dihedrals = []
        phi0_dihedrals = []
        for x in dihedrals_values_article_sin:
            K_dihedrals.append(kBT*np.sqrt(dihedrals_values_article_sin[x]*dihedrals_values_article_sin[x] + dihedrals_values_article_cos[x]*dihedrals_values_article_cos[x]))
            phi0_dihedrals.append(np.arctan2(dihedrals_values_article_sin[x], dihedrals_values_article_cos[x]))

        # Non-bonded energy parameters
        epsilonRepulsive = 0.581685 ### Kcal/mol, kBT at 293 K
        epsilonAttractive = 0.52*1.485*epsilonRepulsive
        #cutOffLJ = 20.0 ### 2.0 nm
        cutOffLJ = 22.0 ### 2.0 nm
        cutOffDH = 40.0 ### 4 nm
        skin = 4.0 ### 0.4 nm
        cutOffVerletFactor = max(cutOffLJ, cutOffDH)
        cutOffVerletFactor = (cutOffVerletFactor + skin)/cutOffVerletFactor
        self.logger.info(f"Dielectric = {dielectricConstant}, epsilonRepulsive = {epsilonRepulsive}, epsilonAttractive = {epsilonAttractive}")
        self.logger.info(f"cutOffLJ = {cutOffLJ}, cutOffDH = {cutOffDH}")
        
        ############################################################
        ######################  Import files  ######################
        ############################################################
        if domains_file != None:
            if coordinates_file == None:
                self.logger.error(f"[IDP_FRET] If domains are provided, a coordinates file must be also present")
                raise Exception("[IDP_FRET] Coordinates file not provided")

        if interfaces_file_A != None or interfaces_file_B != None:
            if coordinates_file == None:
                self.logger.error(f"[IDP_FRET] If interfaces are provided, a coordinates file must be also present")
                raise Exception("[IDP_FRET] Coordinates file not provided")

        if interfaces_file_A != None and interfaces_file_B == None:
            self.logger.error(f"[IDP_FRET] Either both interfaces files are provided, or none of them")
            raise Exception("[IDP_FRET] Interface file not provided")

        if interfaces_file_A == None and interfaces_file_B != None:
            self.logger.error(f"[IDP_FRET] Either both interfaces files are provided, or none of them")
            raise Exception("[IDP_FRET] Interface file not provided")

        if coordinates_file != None:
            if not os.path.exists(coordinates_file):
                self.logger.error(f"[IDP_FRET] Coordinates file not found (selected: {coordinates_file})")
                raise Exception("[IDP_FRET] File not found")
        if bonds_file != None:
            if not os.path.exists(bonds_file):
                self.logger.error(f"[IDP_FRET] Bonds file not found (selected: {bonds_file})")
                raise Exception("[IDP_FRET] File not found")
        if domains_file != None:
            if not os.path.exists(domains_file):
                self.logger.error(f"[IDP_FRET] Domains file not found (selected: {domains_file})")
                raise Exception("[IDP_FRET] File not found")
        if lambdaGroupsFile != None:
            if not os.path.exists(lambdaGroupsFile):
                self.logger.error(f"[IDP_FRET] lambdaGroups file not found (selected: {lambdaGroupsFile})")
                raise Exception("[IDP_FRET] lambdaGroups file not provided")
        if lambdaSoft != 1.0:
            if lambdaGroupsFile == None:
                self.logger.error(f"[IDP_FRET] if lambdaSoft != 1.0 you must provide lambdaGroupsFile")
                raise Exception("[IDP_FRET] lambdaGroups file not provided")

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
            self.logger.error("[IDP_FRET] Number of coordinates ({:d}) does not match sequence length ({:d})".format(len(coordinates), len(sequence)))
            raise Exception("[IDP_FRET] Mismatch between coordinates file and sequence")
        
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
                            self.logger.error("[IDP_FRET] Bonds file {:s}, error on line {:d}: with one field is present it is assumed a range to be present, but '-' was not detected".format(bonds_file, iline+1))
                            raise Exception("[IDP_FRET] Wrong format in bonds file")
                    elif len(line_form) == 2: ### single bond expected
                        if re.search(r'\d+-\d+', line_form[0]) or re.search(r'\d+-\d+', line_form[1]):
                            self.logger.error("[IDP_FRET] Bonds file {:s}, error on line {:d}: with two fields present it is assumed a single bond to be listed, but a range was instead detected".format(bonds_file, iline+1))
                            raise Exception("[IDP_FRET] Wrong format in bonds file")
                        bonds.append([int(line_form[0]), int(line_form[1])])
                    else:
                        self.logger.error("[IDP_FRET] Bonds file {:s}, error on line {:d}: invalid input. Only one field is allowed for a range (e.g. 4-7) or two fields for a single bond (e.g. 4 7)".format(bonds_file, iline+1))
                        raise Exception("[IDP_FRET] Wrong format in bonds file")
        else:
            for i in range(0,len(sequence)-1):
                bonds.append([i, i+1])
        
        ### import angles
        angles = []
        if angles_file != None:
            with open(angles_file, 'r') as file:
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
                            for i in range(i1, i2-1):
                                angles.append([i, i+1, i+2])
                        else:
                            self.logger.error("[IDP_FRET] Angles file {:s}, error on line {:d}: with one field is present it is assumed a range to be present, but '-' was not detected".format(angles_file, iline+1))
                            raise Exception("[IDP_FRET] Wrong format in angles file")
                    elif len(line_form) == 3: ### single angle expected
                        if re.search(r'\d+-\d+', line_form[0]) or re.search(r'\d+-\d+', line_form[1]) or re.search(r'\d+-\d+', line_form[2]):
                            self.logger.error("[IDP_FRET] Angles file {:s}, error on line {:d}: with 3 fields present it is assumed a single angle to be listed, but a range was instead detected".format(angles_file, iline+1))
                            raise Exception("[IDP_FRET] Wrong format in angles file")
                        angles.append([int(line_form[0]), int(line_form[1]), int(line_form[2])])
                    else:
                        self.logger.error("[IDP_FRET] Angles file {:s}, error on line {:d}: invalid input. Only one field is allowed for a range (e.g. 4-7) or 3 fields for a single angle (e.g. 4 7 11)".format(angles_file, iline+1))
                        raise Exception("[IDP_FRET] Wrong format in angles file")
        else:
            for i in range(0,len(sequence)-2):
                angles.append([i, i+1, i+2])

        ### import dihedrals
        dihedrals = []
        if dihedrals_file != None:
            with open(dihedrals_file, 'r') as file:
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
                            for i in range(i1, i2-2):
                                dihedrals.append([i, i+1, i+2, i+3])
                        else:
                            self.logger.error("[IDP_FRET] Dihedrals file {:s}, error on line {:d}: with one field is present it is assumed a range to be present, but '-' was not detected".format(dihedrals_file, iline+1))
                            raise Exception("[IDP_FRET] Wrong format in dihedrals file")
                    elif len(line_form) == 4: ### single dihedral expected
                        if re.search(r'\d+-\d+', line_form[0]) or re.search(r'\d+-\d+', line_form[1]) or re.search(r'\d+-\d+', line_form[2]) or re.search(r'\d+-\d+', line_form[3]):
                            self.logger.error("[IDP_FRET] Dihedrals file {:s}, error on line {:d}: with 4 fields present it is assumed a single angle to be listed, but a range was instead detected".format(dihedrals_file, iline+1))
                            raise Exception("[IDP_FRET] Wrong format in dihedrals file")
                        dihedrals.append([int(line_form[0]), int(line_form[1]), int(line_form[2]), int(line_form[3])])
                    else:
                        self.logger.error("[IDP_FRET] Dihedrals file {:s}, error on line {:d}: invalid input. Only one field is allowed for a range (e.g. 4-7) or 4 fields for a single angle (e.g. 4 7 11 6)".format(dihedrals_file, iline+1))
                        raise Exception("[IDP_FRET] Wrong format in dihedrals file")
        else:
            for i in range(0,len(sequence)-3):
                dihedrals.append([i, i+1, i+2, i+3])

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
            self.logger.error("[IDP_FRET] Interfaces files must have the same length")
            raise Exception("[IDP_FRET] Incompatible interfaces files")
        
        ### import lambdaGroups
        if lambdaGroupsFile != None:
            lambdaGroups = []
            with open(lambdaGroupsFile, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    line_form = line.rsplit()
                    lambdaGroup_local = []
                    for field in line_form:
                        is_range = re.search(r'\d+-\d+', field)
                        if is_range:  ## detect presence of range
                            i1 = int(field.rsplit('-')[0])
                            i2 = int(field.rsplit('-')[1])
                            for i in range(i1, i2+1):
                                lambdaGroup_local.append(i)
                        else:
                            lambdaGroup_local.append(int(field))
                    lambdaGroups.append(lambdaGroup_local)
            if len(lambdaGroups) != 2:
                self.logger.error("[IDP_FRET] lambdaGroups files must have exactly two lines (one per group)")
                raise Exception("[IDP_FRET] Wrong format for lambdaGroupFile")

        ############################################################
        ######################  Set up model  ######################
        ############################################################

        units = self.getUnits()

        if units.getName() != "KcalMol_A":
            self.logger.error(f"[IDP_FRET] Units are not set correctly. Please set units to \"KcalMol_A\" (selected: {units.getName()})")
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

        ### Add amino acid Z: NOH, corresponding to average mass, radius and sigma, zero charge and zero hydrophobicity
        aminoacids_3to1['Z'] = 'NOH'
        names['NOH'] = 'Z'
        charges['NOH'] = 0
        masses['NOH'] = np.mean([masses[x] for x in masses])
        radii['NOH'] = np.mean([radii[x] for x in radii])
        ### Add amino acid X: AVE, corresponding to average mass, radius and sigma, zero charge and average hydrophobicity
        aminoacids_3to1['X'] = 'AVE'
        names['AVE'] = 'X'
        charges['AVE'] = 0
        masses['AVE'] = np.mean([masses[x] for x in masses])
        radii['AVE'] = np.mean([radii[x] for x in radii])

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
                structure["data"].append([i,aminoacids_3to1[s], -1])
            except:
                self.logger.error(f"[IDP_FRET] Aminoacid {s} not recognized")
                raise Exception("Aminoacid not recognized")

        ### domains: first we assign the modelId for each domain
        domainCounter = -1
        for idomain in range(len(domains)):
            domainCounter += 1
            for i in domains[idomain]:
                structure["data"][i][2] = domainCounter
        ### domains: then, we assign a modelID to each bead in the disordered part
        for i in range(len(sequence)):
            if structure["data"][i][2] == -1:
                domainCounter += 1
                structure["data"][i][2] = domainCounter

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

        # chain angles
        forcefield["angles"] = {}
        forcefield["angles"]["type"]             = ["Bond3","HarmonicAngularCommon_K_theta0"]
        forcefield["angles"]["parameters"]       = {}
        forcefield["angles"]["parameters"]["K"]  = Ka
        forcefield["angles"]["parameters"]["theta0"] = theta0
        forcefield["angles"]["labels"]           = ["id_i","id_j","id_k"]
        forcefield["angles"]["data"]             = []

        for iang in range(len(angles)):
            ang = angles[iang]
            i = int(ang[0])
            j = int(ang[1])
            k = int(ang[2])
            forcefield["angles"]["data"].append([i,j,k])
            exclusions.setdefault(i,[]).append(j)
            exclusions.setdefault(i,[]).append(k)
            exclusions.setdefault(j,[]).append(i)
            exclusions.setdefault(j,[]).append(k)
            exclusions.setdefault(k,[]).append(i)
            exclusions.setdefault(k,[]).append(j)

        # chain dihedrals
        forcefield["dihedrals"] = {}
        forcefield["dihedrals"]["type"]             = ["Bond4","Dihedral4"]
        forcefield["dihedrals"]["parameters"]       = {}
        forcefield["dihedrals"]["labels"]           = ["id_i","id_j","id_k", "id_l", "K", "phi0"]
        forcefield["dihedrals"]["data"]             = []

        for idih in range(len(dihedrals)):
            dih = dihedrals[idih]
            i = int(dih[0])
            j = int(dih[1])
            k = int(dih[2])
            l = int(dih[3])
            forcefield["dihedrals"]["data"].append([i,j,k,l,K_dihedrals,phi0_dihedrals])
            exclusions.setdefault(i,[]).append(j)
            exclusions.setdefault(i,[]).append(k)
            exclusions.setdefault(i,[]).append(l)
            exclusions.setdefault(j,[]).append(i)
            exclusions.setdefault(j,[]).append(k)
            exclusions.setdefault(j,[]).append(l)
            exclusions.setdefault(k,[]).append(i)
            exclusions.setdefault(k,[]).append(j)
            exclusions.setdefault(k,[]).append(l)
            exclusions.setdefault(l,[]).append(i)
            exclusions.setdefault(l,[]).append(j)
            exclusions.setdefault(l,[]).append(k)

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
                        if distance <= 9.0:
                            forcefield["ENM"]["data"].append([i, j, KENM, distance])

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
                        if distance <= 9.0:
                            forcefield["Interfaces"]["data"].append([i, j, KENM, distance])
                            exclusions.setdefault(i,[]).append(j)
                            exclusions.setdefault(j,[]).append(i)

        #NL
        for exc in exclusions:
            exclusions[exc] = sorted(list(set(exclusions[exc])))
        forcefield["nl"] = {}
        forcefield["nl"]["type"]       = ["VerletConditionalListSet", "nonExclIntra_nonExclInter"]
        forcefield["nl"]["parameters"] = {"cutOffVerletFactor": cutOffVerletFactor}
        forcefield["nl"]["labels"]     = ["id", "id_list"]
        forcefield["nl"]["data"] = []

        for i in range(len(sequence)):
            forcefield["nl"]["data"].append([i,exclusions[i]])

        # HYDROPHOBIC
        sigma_max = np.max([hydrophobicity[hydrophobicityScale][x]["sigma"] for x in hydrophobicity[hydrophobicityScale]])
        forcefield["hydrophobic"] = {}
        forcefield["hydrophobic"]["type"]       = ["NonBonded", "SplitLennardJones"]
        forcefield["hydrophobic"]["parameters"] = {"cutOffFactor": cutOffLJ/sigma_max,
                                                   "epsilon_r": epsilonRepulsive,
                                                   "epsilon_a": epsilonAttractive,
                                                   "condition":"inter"}
        forcefield["hydrophobic"]["labels"] = ["name_i", "name_j", "epsilon", "sigma"]
        forcefield["hydrophobic"]["data"]   = []

        for t1 in names:
            for t2 in names:
                eps = np.sqrt(hydrophobicity[hydrophobicityScale][t1]["lambda"]*hydrophobicity[hydrophobicityScale][t2]["lambda"])
                sigma = 0.5*(hydrophobicity[hydrophobicityScale][t1]["sigma"] + hydrophobicity[hydrophobicityScale][t2]["sigma"])
                forcefield["hydrophobic"]["data"].append([t1,t2,eps,sigma])

        #DH
        forcefield["DH"] = {}
        forcefield["DH"]["type"]       = ["NonBonded", "DebyeHuckel"]
        forcefield["DH"]["parameters"] = {"cutOffFactor": cutOffDH/debyeLength,
                                          "debyeLength": debyeLength,
                                          "dielectricConstant": dielectricConstant,
                                          "condition":"inter"}
        
        ###### ADD SOFT POTENTIAL FOR SELECTED GROUPS, IF SPECIFIED
        if lambdaGroupsFile != None:
            forcefield["nl"]["type"]       = ["VerletConditionalListSet", "NonExclIdGroup1Intra_NonExclIdGroup2Intra_NonExclInter_NonExclNoGroup"]
            forcefield["nl"]["parameters"]["idGroup1"] = lambdaGroups[0]
            forcefield["nl"]["parameters"]["idGroup2"] = lambdaGroups[1]
            forcefield["hydrophobic"]["parameters"]["condition"] = "interModels" 
            forcefield["DH"]["parameters"]["condition"] = "interModels" 

            forcefield["softHydrophobic"] = {}
            forcefield["softHydrophobic"]["type"]       = ["NonBonded", "softSplitLennardJones"]
            forcefield["softHydrophobic"]["parameters"] = {
                    "cutOffFactor": cutOffLJ/sigma_max,
                    "epsilon_r"   : epsilonRepulsive,
                    "epsilon_a"   : epsilonAttractive,
                    "lambda"      : lambdaSoft,
                    "alpha"       : alphaSoft,
                    "n"           : nSoft,
                    "condition"   :"interGroups"}
            forcefield["softHydrophobic"]["labels"] = ["name_i", "name_j", "epsilon", "sigma"]
            forcefield["softHydrophobic"]["data"]   = []
            for t1 in names:
                for t2 in names:
                    eps = np.sqrt(hydrophobicity[hydrophobicityScale][t1]["lambda"]*hydrophobicity[hydrophobicityScale][t2]["lambda"])
                    sigma = 0.5*(hydrophobicity[hydrophobicityScale][t1]["sigma"] + hydrophobicity[hydrophobicityScale][t2]["sigma"])
                    forcefield["softHydrophobic"]["data"].append([t1,t2,eps,sigma])
            
            forcefield["softDH"] = {}
            forcefield["softDH"]["type"]       = ["NonBonded", "softDH"]
            forcefield["softDH"]["parameters"] = {
                    "cutOffFactor"      : cutOffDH/debyeLength,
                    "debyeLength"       : debyeLength,
                    "dielectricConstant": dielectricConstant,
                    "lambda"            : lambdaSoft,
                    "condition"         :"interGroups"}


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
