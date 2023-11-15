import os
import subprocess
import datetime
import time
import xml.etree.ElementTree as ET
import xmltodict
import tempfile
import numpy as np

from tricorder.mri.data.nifti import FileSave

class EvaluateSegmentationWrapper:
    def __init__(self, bin_file, useAll=True, metrics="DICE,PROBDST", threshold=None, useMMunit=False, mode=0, return_rawdict=False) -> None:
        r""" List of available metrics:- (if useAll is set to True, then all will be executed. If it is False, then the coma-seperated string "metrics" will be used)
            # DICE: Dice Coefficient
            # JACRD: Jaccard Coefficient
            # GCOERR: Global Consistency Error
            # VOLSMTY: Volumetric Similarity Coefficient
            # KAPPA: Cohen Kappa
            # AUC: Area under ROC Curve (one system state)
            # RNDIND: Rand Index
            # ADJRIND: Adjusted Rand Index
            # ICCORR: Interclass Correlation
            # MUTINF: Mutual Information
            # FALLOUT: Fallout (false positive rate)
            # COEFVAR: Coefficient of Variation
            # AVGDIST: Average Hausdorff Distance (in voxel or millimeter according to -unit)
            # bAVD: Balanced Average Hausdorff Distance
            # HDRFDST: Hausdorff Distance in voxels HDRFDST@0.95@ means use 0.95 quantile to avoid outliers. Default is quantile of 1 which means exact Hausdorff distance  (in voxel or millimeter according to -unit)
            # VARINFO: Variation of Information
            # PROBDST: Probabilistic Distance
            # MAHLNBS: Mahanabolis Distance
            # SNSVTY: Sensitivity (Recall, true positive rate)
            # SPCFTY: Specificity (true negative rate)
            # PRCISON: Precision
            # FMEASR: F-Measure  FMEASR@0.5@ means use 0.5 as a value for beta in the F-Measure
            # ACURCY: Accuracy
            # TP: true positives in voxel)
            # TN: true negatives (in voxel)
            # FP: false positives (in voxel)
            # FN: false negatives (in voxel)
            # SEGVOL: segmented volume (in voxel or milliliter according to -unit)
            # REFVOL: reference volume (in voxel or milliliter according to -unit)

            Example of "metrics" string (will be considered only of useAll is False): DICE,PROBDST,bAVD

            Args:
                bin_file: Location of the binary file of EvaluteSegmentation (can be obtained from https://github.com/Visceral-Project/EvaluateSegmentation/tree/master/builds/Ubuntu).
                          The latest binary file for Ubuntu, EvaluateSegmentation-2020.08.28-Ubuntu.zip, works on other Linux distros as well.
                useAll: Use all the available metrics (default: True).
                metrics: coma-seperated string, containing the list of metrics to be used (only if useAll is False).
                threshold: before evaluation convert fuzzy images to binary using the given threshold.
                useMMunit: use millimeter instead of voxel for distances and volumes (default: False (i.e. voxel)).
                mode: 0,1,or 2 - #0: volume segmentations, #1: landmark localisation, #2: lesion detection (default: 0, the only one which has been tested).
                return_rawdict: return the complete raw metadictonary (the parsed xml file) along with the results. If set to False (default), onlt the results' dict will be returned.
        """
        self.bin_file = bin_file
        self.useAll = useAll
        self.metrics = metrics
        self.threshold = threshold
        self.useMMunit = useMMunit
        self.mode = mode
        self.return_rawdict = return_rawdict

    def parse_params(self, useAll, metrics, threshold, useMMunit, mode, return_rawdict):
        useAll = self.useAll if useAll == -1 else useAll
        metrics = self.metrics if metrics == -1 else metrics
        threshold = self.threshold if threshold == -1 else threshold
        useMMunit = self.useMMunit if useMMunit == -1 else useMMunit 
        mode = self.mode if mode == -1 else mode 
        return_rawdict = self.return_rawdict if return_rawdict == -1 else return_rawdict 
        return (useAll, metrics, threshold, useMMunit, mode, return_rawdict)

    def get_scores(self, truth, prediction, xmlpath=None, useAll=-1, metrics=-1, threshold=-1, useMMunit=-1, mode=-1, return_rawdict=-1):
        r"""
        Args:
            truth: Path to the ground-truth file (Nifti) or a numpy array containing the ground-truth.
            prediction: Path to the prediction file (Nifti) or a numpy array containing the ground-truth.
            xmlpath: If it is desired to save the actual output of the EvaluateSegmentation program as an XML file (default: None).
            useAll, metrics, threshold, useMMunit, mode, return_rawdict: Same as the constructor. They are to be supplied to overwrite the default values set earlier. 
                                                                         If these values are supplied, the ones set in the class will be ignored.   
        """
        useAll, metrics, threshold, useMMunit, mode, return_rawdict = self.parse_params(useAll, metrics, threshold, useMMunit, mode, return_rawdict)

        start_time = time.time()

        #If arrays are supplied instead of paths, saving them as temp files before continuing
        if type(truth) != str:
            tmpTrue = tempfile.NamedTemporaryFile(suffix=".nii.gz")
            FileSave(truth, tmpTrue.name, useV2=False)
            truth = tmpTrue.name

        if type(prediction) != str:
            tmpPred = tempfile.NamedTemporaryFile(suffix=".nii.gz")
            FileSave(prediction, tmpPred.name, useV2=False)
            prediction = tmpPred.name

        cmd = [self.bin_file, truth, prediction]

        #Only mode 0 has been tested
        if mode == 1:
            cmd.insert(1, "-loc")
        elif mode == 2:
            cmd.insert(1, "-det")

        if threshold is not None:
            cmd += ["-thd", str(threshold)]
        if xmlpath is not None:
            cmd += ["-xml", xmlpath]
        else:
            tmpXML = tempfile.NamedTemporaryFile(suffix=".xml")
            cmd += ["-xml", tmpXML.name]
        cmd += ["-unit", "millimeter" if useMMunit else "voxel"]
        cmd += ["-use", "all" if useAll else metrics]        

        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()

        print("\n\n--- %s seconds ---" % (time.time() - start_time))

        try:
            tmpTrue.close()
        except:
            pass
        try:
            tmpPred.close()
        except:
            pass

        if xmlpath is not None:
            data = xmltodict.parse(ET.tostring(ET.parse(xmlpath).getroot(), encoding='utf8', method='xml'))
        else:
            data = xmltodict.parse(ET.tostring(ET.parse(tmpXML.name).getroot(), encoding='utf8', method='xml'))
            tmpXML.close()
        
        results = {}
        for (k,v) in data["measurement"]["metrics"].items():
            results[k] = float(v["@value"])

        if return_rawdict:
            return results, data
        else:
            return results