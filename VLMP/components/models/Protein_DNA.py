import sys, os
import copy
import json

import logging

import itertools

from VLMP.components.models import modelBase
from VLMP.utils.input import getSubParameters

import math
import importlib

import pyUAMMD
from pyUAMMD.utils.merging.merging import mergeSimulationsSet

class Protein_DNA(modelBase):
    """
    {"author": "Pablo Ibáñez-Freire, Salvatore Assenza",
     "description":
     "A model that combines proteins and DNA. There are several models available for each
     type of component. Currently:
     Protein: Calvados
     DNA: MADna (Assenza2022)
      <p>
     "parameters":{
        "proteinModel":{"description":"Protein model to be employed. At the moment, only 'Calvados' is available.",
                    "type":"str"},
        "dnaModel":{"description":"DNA model to be employed. At the moment, only 'MADna' is available.",
                    "type":"str"},
        "proteinParameters":{"description":"Dictionary containing all the parameters related to the protein model.",
                       "type":"dictionary"},
        "dnaParameters":{"description":"Dictionary containing all the parameters related to the DNA model.",
                       "type":"dictionary"},
        "dnaProteinInteraction":{"description": "Interaction between DNA and protein, containing Debye-Hückel and excluded volume. Available options are 'hard' for standard WCA or 'sof' for a soft-core potential to gently remove overlaps.",
                       "type":"str"}
     },
     "example":"
         {
            "type":"Protein_DNA",
            "parameters":{
                "proteinModel": "Calvados",
                "proteinParameters": {"sequence": "GKGGAKRHRKVLRDNIQGITKPAIRRLARRGGVKRISGLIYEE", "ionicStrength": 150.0, "pH": 7.0, "version": 3}
                "dnaModel": "MADna",
                "dnaParameters": {"sequence":"ATCGGATCCGAT", "ionicStrength": 150.0},
                "dnaProteinInteraction": "hard"
            }
         }
        ",
     "references":[
         ".. [Assenza2022] Assenza, S., & Pérez, R. (2022). Accurate Sequence-Dependent Coarse-Grained Model for Conformational and Elastic Properties of Double-Stranded DNA. Journal of Chemical Theory and Computation, 18(5), 3239-3256."
     ]
    }
    """

    availableParameters = {"proteinModel","dnaModel",
                           "proteinParameters","dnaParameters", "dnaProteinInteraction", "lambda"}
    requiredParameters = {"proteinModel","dnaModel",
                           "proteinParameters","dnaParameters", "dnaProteinInteraction"}
    definedSelections   = {"DNA","PROTEIN"}


    def __init__(self,name,**params):
        super().__init__(_type = self.__class__.__name__,
                         _name = name,
                         availableParameters = self.availableParameters,
                         requiredParameters  = self.requiredParameters,
                         definedSelections   = self.definedSelections,
                         **params)

        ############################################################
        ############################################################
        ############################################################

        units = self.getUnits()

        if units.getName() != "KcalMol_A":
            self.logger.error(f"[Protein_DNA] Units are not set correctly. Please set units to \"KcalMol_A\" (selected: {units.getName()})")
            raise Exception("Not correct units")

        ############################################################
        #################   INITIALIZE PARAMETERS   ################
        ############################################################

        self.dnaSequence = params.get("dnaSequence")
        self.proteinSequence = params.get("proteinSequence")

        self.dnaModel = params.get("dnaModel")
        self.proteinModel = params.get("proteinModel")

        self.proteinParameters = params.get("proteinParameters")
        self.dnaParameters = params.get("dnaParameters")
        
        self.dnaProteinInteraction = params.get("dnaProteinInteraction")
        if self.dnaProteinInteraction != 'hard':
            self.dnaProteinInteractionLambda = params.get("lambda")

        ############################################################
        ######################   LOAD MODELS   #####################
        ############################################################

        proteinModule = importlib.import_module('VLMP.components.models.' + self.proteinModel) # import dynamically the library
        #proteinModule = importlib.import_module('working_VLMP.models.' + self.proteinModel) # import dynamically the library
        func_name = getattr(proteinModule, self.proteinModel)
        protein = func_name(name = f"PROTEIN", units = self.getUnits(), types = self.getTypes(), ensemble = self.getEnsemble(), **self.proteinParameters).getSimulation()
        dnaModule = importlib.import_module('VLMP.components.models.' + self.dnaModel) # import dynamically the library
        func_name = getattr(dnaModule, self.dnaModel)
        dna = func_name(name = f"DNA", units = self.getUnits(), types = self.getTypes(), ensemble = self.getEnsemble(), **self.dnaParameters).getSimulation()
        
        ############################################################
        ###############   ADD TYPES FOR SELECTIONS   ###############
        ############################################################

        self.proteinTypes = []
        typeIndex = protein["topology"]["structure"]["labels"].index("type")
        for part in protein["topology"]["structure"]["data"]:
            self.proteinTypes.append(part[typeIndex])
        self.proteinTypes = list(set(self.proteinTypes))

        self.dnaTypes = []
        typeIndex = dna["topology"]["structure"]["labels"].index("type")
        for part in dna["topology"]["structure"]["data"]:
            self.dnaTypes.append(part[typeIndex])
        self.dnaTypes = list(set(self.dnaTypes))

        ############################################################
        ###################   MERGE FORCEFIELDS   ##################
        ############################################################
        for x in list(protein['topology']['forceField'].keys()):
            protein['topology']['forceField'][x+'_PROTEIN'] = protein['topology']['forceField'].pop(x) ### change name of keys in protein
        for x in list(dna['topology']['forceField'].keys()):
            dna['topology']['forceField'][x+'_DNA'] = dna['topology']['forceField'].pop(x) ### change name of keys in dna
        
        # Combine the two models
        sim = pyUAMMD.simulation()
        sim.append(protein,mode="modelId")
        sim.append(dna,mode="modelId")
        sim['topology']['forceField']['groups'] = {}
        sim['topology']['forceField']['groups']['type'] = ['Groups', 'GroupsList']
        sim['topology']['forceField']['groups']['labels'] = ['name', 'type', 'selection']
        sim['topology']['forceField']['groups']['data'] = []
        sim['topology']['forceField']['groups']['data'].append(['dna', 'Types', self.dnaTypes])
        sim['topology']['forceField']['groups']['data'].append(['protein', 'Types', self.proteinTypes])

        # Debye-Hückel
        sim['topology']['forceField']['DH_INTRA'] = sim['topology']['forceField'].pop('DH_DNA')
        sim['topology']['forceField']['DH_INTRA']['parameters']['condition'] = 'intra'
        if 'DH_PROTEIN' in sim['topology']['forceField']:
            sim['topology']['forceField'].pop('DH_PROTEIN')

        # For other nonbonded interactions, impose that they act only on the right group
        for x in sim['topology']['forceField'].keys():
            if sim['topology']['forceField'][x]['type'][0] == 'NonBonded' and x != 'DH_INTRA':
                if x.endswith('_DNA'):
                    sim['topology']['forceField'][x]['parameters']['group'] = 'dna'
                    sim['topology']['forceField'][x]['parameters']['condition'] = 'intra'
                if x.endswith('_PROTEIN'):
                    sim['topology']['forceField'][x]['parameters']['group'] = 'protein'
                    sim['topology']['forceField'][x]['parameters']['condition'] = 'intra'

        # Merge exclusion lists and remove the separated ones
        sim['topology']['forceField']['nl'] = copy.deepcopy(sim['topology']['forceField']['nl_PROTEIN'])
        sim['topology']['forceField']['nl']['data'].extend(sim['topology']['forceField']['nl_DNA']['data'])
        sim['topology']['forceField']['nl']['type'] = ['VerletConditionalListSet','nonExclIntra_nonExclInter']
        sim['topology']['forceField'].pop('nl_PROTEIN')
        sim['topology']['forceField'].pop('nl_DNA')

        ############################################################
        ################   DNA-PROTEIN INTERACTION   ###############
        ############################################################
        
        dielectricConstant = sim['topology']['forceField']['DH_INTRA']['parameters']['dielectricConstant']
        debyeLength = sim['topology']['forceField']['DH_INTRA']['parameters']['debyeLength']
        cutOffFactor = sim['topology']['forceField']['DH_INTRA']['parameters']['cutOffFactor']
        sim["topology"]["forceField"]["inter_WCA_DH"] = {}
        sim["topology"]["forceField"]["inter_WCA_DH"]["parameters"] = {}
        if self.dnaProteinInteraction == 'hard':
            sim["topology"]["forceField"]["inter_WCA_DH"]["type"] = ["NonBonded","WCA_DH"]
        elif self.dnaProteinInteraction == 'soft':
            sim["topology"]["forceField"]["inter_WCA_DH"]["type"] = ["NonBonded","softWCA_DH"]
            sim["topology"]["forceField"]["inter_WCA_DH"]["parameters"]["alpha"] = 1.0
            sim["topology"]["forceField"]["inter_WCA_DH"]["parameters"]["n"] = 1
            sim["topology"]["forceField"]["inter_WCA_DH"]["parameters"]["lambda"] = self.dnaProteinInteractionLambda
        else:
            self.logger.error("[Protein_DNA] '{:s}' is not a valid option for dnaProteinInteraction parameter. Available choices: 'hard' or 'soft'".format(self.dnaProteinInteraction))
            raise Exception("Wrong parameter")
        sim["topology"]["forceField"]["inter_WCA_DH"]["parameters"]["condition"] = "inter"
        sim["topology"]["forceField"]["inter_WCA_DH"]["parameters"]["dielectricConstant"] = dielectricConstant
        sim["topology"]["forceField"]["inter_WCA_DH"]["parameters"]["debyeLength"] = debyeLength
        sim["topology"]["forceField"]["inter_WCA_DH"]["parameters"]["cutOffFactor"] = cutOffFactor
        sim["topology"]["forceField"]["inter_WCA_DH"]["labels"] = ["name_i","name_j","epsilon","sigma","chargeProduct"]
        eps_WCA = 1.0
        tipi = copy.deepcopy(self.getTypes().getTypes())
        tipi['P']['charge'] = -1.0 ## charge of phosphate for interaction with protein is -1.0
        sim["topology"]["forceField"]["inter_WCA_DH"]["data"] = []
        for t1 in tipi:
            for t2 in tipi:
                sigma_WCA = tipi[t1]['radius'] + tipi[t2]['radius']
                chargeProduct = tipi[t1]['charge']*tipi[t2]['charge']
                sim["topology"]["forceField"]["inter_WCA_DH"]["data"].append([tipi[t1]['name'], tipi[t2]['name'], eps_WCA, sigma_WCA, chargeProduct])


        ############################################################
        #######################   FINALIZE   #######################
        ############################################################

        self.setState(copy.deepcopy(sim["state"]))
        self.setStructure(copy.deepcopy(sim["topology"]["structure"]))
        self.setForceField(copy.deepcopy(sim["topology"]["forceField"]))


    def processSelection(self,selectionType,selectionOptions):

        structure = self.getStructure()
        idIndex   = structure["labels"].index("id")
        typeIndex = structure["labels"].index("type")

        if selectionType == "DNA":
            sel = set()
            for dnaPart in structure["data"]:
                if dnaPart[typeIndex] in self.dnaTypes:
                    sel.add(dnaPart[idIndex])
            return sel

        if selectionType == "PROTEIN":
            sel = set()
            for protPart in structure["data"]:
                if protPart[typeIndex] in self.proteinTypes:
                    sel.add(protPart[idIndex])
            return sel

        return None
