import sys, os

import copy

import logging

import numpy as np
import networkx as nx
import random
import json

from VLMP.components.models import modelBase

from pyGrained.utils.data import getData
import re

class RnaSecondary(modelBase):
    """
    Component name: RnaSecondary
    Component type: model

    Author: Salvatore Assenza
    Date: 13/02/2025

    Model for RNA based on secondary structure
    The input is a json file as obtained by the forna webserver (http://rna.tbi.univie.ac.at/forna/)
    The (important part of the)structure is the following, in case you want to create the json manually:
    - modelName:
         - internalBizarreCode:
             - dotbracket
             - nodes:
                 - 0:
                    - x
                    - y
                - 1:
                    - x
                    - y
                ...

    """

    def __init__(self,name,**params):
        super().__init__(_type = self.__class__.__name__,
                         _name= name,
                         availableParameters = {"jSon", "stepDsRna", "stepSsRna", "sigmaDsRna", "sigmaSsRna", "lpDsRna", "lpSsRna", "isSoft", "psfFile"},
                         requiredParameters  = {"jSon"},
                         definedSelections   = {"particleId"},
                         **params)

        ############################################################

        jSon = params.get("jSon")
        if not os.path.exists(jSon):
            self.logger.error("[RnaSecondary] Json file {:s} does not exist!".format(jSon))
            raise Exception("[RnaSecondary] Missing input file")
        with open(jSon, 'r') as file:
            infoMolecule = json.load(file)

        keycombo = []
        for key1 in infoMolecule:
            for key2 in infoMolecule[key1]:
                for key3 in infoMolecule[key1][key2]:
                    if key3 == 'dotbracket':
                        keycombo = [key1, key2]
        if keycombo == []:
            self.logger.error("[RnaSecondary] Json structure is not in the expected form. Unable to find key 'dotbracket'")
            raise Exception("[RnaSecondary] Bad input file")
        infoMolecule = infoMolecule[keycombo[0]][keycombo[1]]
        secondaryStructure = infoMolecule['dotbracket']

        stepDsRna = params.get("stepDsRna", 2.9)
        stepSsRna = params.get("stepDsRna", 4.91) # PNAS 109,799 2011 (MEMO: check latest Ritort or thesis in Borja's group)
        sigmaDsRna = params.get("sigmaDsRna", 24.0)
        sigmaSsRna = params.get("sigmaDsRna", 10.0)
        lpDsRna = params.get("lpDsRna", 670.0)
        lpSsRna = params.get("lpSsRna", 20.0)
        isSoft = params.get("isSoft", False)
        psfFile = params.get("psfFile", False)

        #self.logger.info(f"Dielectric = {dielectricConstant}, epsilon = {epsilon}, cutOffLJ = {cutOffLJ}, cutOffDH = {cutOffDH}")

        #############################################
        #############  Build topology  ##############
        #############################################

        # 0. Sanity check on structure and create couples
        open_bracket_list = []
        pair_list = []
        for i in range(len(secondaryStructure)):
            if secondaryStructure[i] == '(':
                open_bracket_list.append(i)
            if secondaryStructure[i] == ')':
                if len(open_bracket_list) == 0:
                    self.logger.error("[RnaSecondary] Provided structure has more closed than open brackets")
                    raise Exception("[RnaSecondary] Bad secondary structure")
                pair_list.append([open_bracket_list[-1], i])
                open_bracket_list.pop(-1)
        if len(open_bracket_list) > 0:
            self.logger.error("[RnaSecondary] Provided structure has more open than closed brackets")
            raise Exception("[RnaSecondary] Bad secondary structure")

        # 1. create structure dictionary
        structureDic = {}
        for i in range(len(secondaryStructure)):
            structureDic[i] = {}
            structureDic[i]['type'] = 'ssRNA'
        for x in pair_list:
            structureDic[x[0]]['type'] = 'dsRNA'
            structureDic[x[0]]['partner'] = x[1]
            structureDic[x[1]]['type'] = 'dsRNA'
            structureDic[x[1]]['partner'] = x[0]
        
        natoms = 0
        for i in range(len(secondaryStructure)):
            newParticle = False
            if 'idCoarseGrained' not in structureDic[i]:
                natoms += 1
                structureDic[i]['idCoarseGrained'] = natoms-1
                if structureDic[i]['type'] == 'dsRNA':
                    partner = structureDic[i]['partner']
                    structureDic[partner]['idCoarseGrained'] = natoms-1

        # 2. create dictionary with coarse-grained model
        cgModelDic = {}
        for icg in range(natoms):
            cgModelDic[icg] = {}
            cgModelDic[icg]['elements'] = []
            cgModelDic[icg]['bondedNeighbors'] = []
        for i in range(len(secondaryStructure)):
            icg = structureDic[i]['idCoarseGrained']
            cgModelDic[icg]['elements'].append(i)
        for icg in range(natoms):
            if len(cgModelDic[icg]['elements']) > 1:
                cgModelDic[icg]['elements'] = sorted(cgModelDic[icg]['elements'])
                cgModelDic[icg]['sigma'] = sigmaDsRna
            else:
                cgModelDic[icg]['sigma'] = sigmaSsRna

        # 3. create list of bonds
        bonds = []
        for i in range(len(secondaryStructure)-1):
            icg = structureDic[i]['idCoarseGrained']
            icg2 = structureDic[i+1]['idCoarseGrained']
            if len(cgModelDic[icg]['elements']) == 1 or len(cgModelDic[icg2]['elements']) == 1: ## step of ssRNA if either bead is ss
                r0 = stepSsRna
            else:
                r0 = stepDsRna
            bondIds = sorted([icg, icg2])
            bondIds.append(r0)
            bonds.append(bondIds)
        seen = set()
        bonds = [x for x in bonds if tuple(x) not in seen and not seen.add(tuple(x))] ## erase duplicates
        bonds = sorted(bonds, key=lambda x:x[0])
        for x in bonds:
            icg = x[0]
            icg2 = x[1]
            cgModelDic[icg]['bondedNeighbors'].append(icg2)
            cgModelDic[icg2]['bondedNeighbors'].append(icg)
        
        # 4. create list of angles
        angles = []
        Tloc = self.getEnsemble().getEnsembleComponent("temperature")
        kBT = self.getUnits().getConstant("KBOLTZ")*Tloc
        def langevinInverse(x):
            fac1 = 3-1.00651*x*x-0.962251*x*x*x*x+1.47353*x*x*x*x*x*x-0.48953*x*x*x*x*x*x*x*x
            return x*fac1/((1.0-x)*(1+1.01524*x))
        kDsRna = kBT*langevinInverse(np.exp(-1.0*stepDsRna/lpDsRna))
        kSsRna = kBT*langevinInverse(np.exp(-1.0*stepSsRna/lpSsRna))
        for icg1 in range(natoms):
            for icg2 in cgModelDic[icg1]['bondedNeighbors']:
                for icg3 in cgModelDic[icg2]['bondedNeighbors']:
                    if icg3 != icg1:
                        angles.append([icg1, icg2, icg3, kDsRna])
        seen = set()
        angles = [x for x in angles if tuple(x) not in seen and not seen.add(tuple(x))] ## erase duplicates
        for iang in range(len(angles)-1, -1, -1):
            x = angles[iang]
            isSS = False
            if len(cgModelDic[x[0]]['elements']) == 1 or len(cgModelDic[x[1]]['elements']) == 1 or len(cgModelDic[x[2]]['elements']) == 1: ## all beads must be dsRNA
                isSS = True
            else: ## more complex case: I have to check the stacking within the triplet
                i0_A = cgModelDic[x[0]]['elements'][0]
                i0_B = cgModelDic[x[0]]['elements'][1]
                i1_A = cgModelDic[x[1]]['elements'][0]
                i1_B = cgModelDic[x[1]]['elements'][1]
                i2_A = cgModelDic[x[2]]['elements'][0]
                i2_B = cgModelDic[x[2]]['elements'][1]
                if i0_A != i1_A-1 or i0_B != i1_B+1: ## no stacking with previous element
                    isSS = True
                if i2_A != i1_A+1 or i2_B != i1_B-1: ## no stacking with next element
                    isSS = True
            if isSS:
                x[3] = kSsRna

        # 5. create list of exclusions by using Dijkstra's algorithm on the weighted network formed by the topology
        G = nx.Graph()
        elist = [tuple(x) for x in bonds]
        G.add_weighted_edges_from(elist)
        len_path = dict(nx.all_pairs_dijkstra(G))
        exclusions = []
        for icg in range(natoms):
            exclusions.append([])
        for icg1 in len_path:
            for icg2 in len_path[icg][0]:
                dmin = 0.5*(cgModelDic[icg1]['sigma'] + cgModelDic[icg2]['sigma'])
                if icg2 != icg1 and len_path[icg1][0][icg2] < dmin:
                    exclusions[icg1].append(icg2)
        exclusions = [sorted(x) for x in exclusions]

        # 6. extract coordinates, renormalize and shift
        coordinates = []
        for icg in range(natoms):
            ilist = cgModelDic[icg]['elements']
            xlist = [infoMolecule['nodes'][i]['x'] for i in ilist]
            ylist = [infoMolecule['nodes'][i]['y'] for i in ilist]
            coordinates.append(np.array([np.mean(xlist), np.mean(ylist)]))
        drForna = np.linalg.norm(coordinates[1] - coordinates[0])
        for xy in coordinates:
            xy *= stepDsRna/drForna
        com = np.mean(coordinates, axis=0)
        for xy in coordinates:
            xy -= com
        coordinates = np.array(coordinates)
        coordinates = np.hstack((coordinates, np.zeros((coordinates.shape[0], 1)))) # add zeros for z coordinate

        # 7. generate psf file, if requested
        if psfFile:
            with open(psfFile, "w") as fichero:
                print("PSF SAVVATORI", file=fichero)
                print("", file=fichero)
                print("       1 !NTITLE", file=fichero)
                print(" REMARKS topology for visualization of RNA secondary structure", file=fichero)
                print("", file=fichero)
                print("{:6d} !NATOM".format(natoms), file=fichero)
                for icg in range(natoms):
                    if len(cgModelDic[icg]['elements']) == 1:
                        tiponome = "DST"
                        tipo = 1
                        atomo = "D"
                    else:
                        tiponome = "SST"
                        tipo = 2
                        atomo = "S"
                    print('{:>8d} {:>4} {:<4d} {:>3} {:>2} {:>7} {:>14.6f} {:>9.4f} {:>11}'.format(icg+1, "A", icg+1, tiponome, atomo, tipo, 0, 1.0, "0"), file=fichero)
                print("", file=fichero)
                print("{:6d} !NBOND: bonds".format(len(bonds)), file=fichero)
                iiter = 0
                itot = 0
                s = ''
                for x in bonds:
                    print(x)
                    iiter += 1
                    itot += 1
                    i1 = x[0]+1
                    i2 = x[1]+1
                    s += '{:>8d}{:>8d}'.format(i1, i2)
                    if iiter == 4:
                        print(s, file=fichero)
                        iiter = 0
                        s = ''
                if len(s)>0:
                    print(s, file=fichero)


        ############################################################
        ######################  Set up model  ######################
        ############################################################
        
        units = self.getUnits()

        ############################################################

        # State

        state = {}
        state["labels"] = ["id","position"]
        state["data"]   = []

        for i in range(len(coordinates)):
            pos = np.array(coordinates[i])
            pos = pos.tolist()
            state["data"].append([i,pos])

        # Structure

        types = self.getTypes()
        types.addType(name = 'ssRNA', mass = 1.0, radius = 0.5*sigmaDsRna, charge = 0.0)
        types.addType(name = 'dsRNA', mass = 1.0, radius = 0.5*sigmaSsRna, charge = 0.0)

        structure = {}
        structure["labels"] = ["id","type", "modelId"]
        structure["data"]   = []

        for icg in range(len(coordinates)):
            if len(cgModelDic[icg]['elements']) == 1:
                tipo = 'ssRNA'
            else:
                tipo = 'dsRNA'
            structure["data"].append([icg, tipo, 0])

        # Forcefield

        forcefield = {}
        exclusions_tmp = {}
        for icg in range(natoms):
            exclusions_tmp[icg] = exclusions[icg]
        exclusions = exclusions_tmp

        # bonds 
        kbonds = 60.0
        forcefield["bonds"] = {}
        forcefield["bonds"]["type"]             = ["Bond2","Harmonic"]
        forcefield["bonds"]["parameters"]       = {}
        forcefield["bonds"]["labels"]           = ["id_i","id_j","K","r0"]
        forcefield["bonds"]["data"]             = []

        for ibnd in range(len(bonds)):
            bnd = bonds[ibnd]
            i = int(bnd[0])
            j = int(bnd[1])
            r0 = float(bnd[2])
            forcefield["bonds"]["data"].append([i,j,kbonds,r0])

        # angles
        forcefield["angles"] = {}
        forcefield["angles"]["type"]             = ["Bond3","KratkyPorod"]
        forcefield["angles"]["parameters"]       = {}
        forcefield["angles"]["labels"]           = ["id_i","id_j","id_k","K"]
        forcefield["angles"]["data"]             = []

        for iang in range(len(angles)):
            ang = angles[iang]
            i1 = int(ang[0])
            i2 = int(ang[1])
            i3 = int(ang[2])
            k = float(ang[3])
            forcefield["angles"]["data"].append([i1,i2,i3,k])

        #NL
        cutOffVerletFactor = 2.0
        for exc in exclusions:
            exclusions[exc] = sorted(list(set(exclusions[exc])))
        forcefield["nl"] = {}
        forcefield["nl"]["type"]       = ["VerletConditionalListSet", "nonExcluded"]
        forcefield["nl"]["parameters"] = {"cutOffVerletFactor": cutOffVerletFactor}
        forcefield["nl"]["labels"]     = ["id", "id_list"]
        forcefield["nl"]["data"] = []

        for i in range(len(coordinates)):
            forcefield["nl"]["data"].append([i,exclusions[i]])

        # WCA
        epsWCA = 1.0
        if isSoft:
            alpha = 1.0
            lambdaPar = self.getEnsemble().getEnsembleComponent("lambda")
            factor = np.power(1-alpha*(1-lambdaPar)*(1-lambdaPar), 1.0/6.0)
            forcefield["softWCA"] = {}
            forcefield["softWCA"]["type"]       = ["NonBonded", "LennardJonesSoftCoreType2"]
            forcefield["softWCA"]["parameters"] = {"cutOffFactor": factor, "condition":"nonExcluded", "alpha":alpha, "n":1}
            forcefield["softWCA"]["labels"] = ["name_i", "name_j", "epsilon", "sigma"]
            forcefield["softWCA"]["data"]   = []
            forcefield["softWCA"]["data"].append(['dsRNA','dsRNA', epsWCA , sigmaDsRna/factor])
            forcefield["softWCA"]["data"].append(['dsRNA','ssRNA', epsWCA , 0.5*(sigmaDsRna+sigmaSsRna)/factor])
            forcefield["softWCA"]["data"].append(['ssRNA','dsRNA', epsWCA , 0.5*(sigmaDsRna+sigmaSsRna)/factor])
            forcefield["softWCA"]["data"].append(['ssRNA','ssRNA', epsWCA , sigmaSsRna/factor])
        else:
            forcefield["WCA"] = {}
            forcefield["WCA"]["type"]       = ["NonBonded", "WCAType2"]
            forcefield["WCA"]["parameters"] = {"cutOffFactor": 2.5*sigmaDsRna, "condition":"nonExcluded"}
            forcefield["WCA"]["labels"] = ["name_i", "name_j", "epsilon", "sigma"]
            forcefield["WCA"]["data"]   = []
            forcefield["WCA"]["data"].append(['dsRNA','dsRNA', epsWCA , sigmaDsRna])
            forcefield["WCA"]["data"].append(['dsRNA','ssRNA', epsWCA , 0.5*(sigmaDsRna+sigmaSsRna)])
            forcefield["WCA"]["data"].append(['ssRNA','dsRNA', epsWCA , 0.5*(sigmaDsRna+sigmaSsRna)])
            forcefield["WCA"]["data"].append(['ssRNA','ssRNA', epsWCA , sigmaSsRna])

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
