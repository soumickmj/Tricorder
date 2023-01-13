import SimpleITK as sitk
import contextlib
import pydicom
import numpy as np
from .sitk import FileSave as stikSave
import os

def __tags2dict(reader, sliceID=0, taginits2ignore=[]):
    tag_dict = {}
    if type(reader) is sitk.ImageSeriesReader: #If it's a series, then the sliceID is needed. Default is 0
        for k in reader.GetMetaDataKeys(sliceID):
            if k.split("|")[0] not in taginits2ignore:
                tag_dict[k] = reader.GetMetaData(sliceID,k)
    else:
        if type(reader) is sitk.ImageSeriesReader:
            for k in reader.GetMetaDataKeys():
                if k.split("|")[0] not in taginits2ignore:
                    tag_dict[k] = reader.GetMetaData(k)
    return tag_dict

def ReadSeries(folder_path, returnIDs=False, return_meta=False, series_ids=None, series2array=True, taginits2ignore=[]):
    if not bool(series_ids):
        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(folder_path)
    if not isinstance(series_ids, list):    
        series_ids = [series_ids]
    imgs = []
    metas = []
    sIDs = []
    for sid in series_ids:
        #read the dicom series

        with contextlib.suppress(Exception):
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(folder_path, sid)
            reader.SetFileNames(dicom_names)
            if return_meta:
                reader.MetaDataDictionaryArrayUpdateOn()
                reader.LoadPrivateTagsOn()
            image = reader.Execute()
            imgs.append(sitk.GetArrayFromImage(image))
            sIDs.append(sid)
            if return_meta:
                metas.append(__tags2dict(reader, taginits2ignore=taginits2ignore))

    if series2array:
        imgs = np.array(imgs)
    if return_meta and returnIDs:
        return imgs, sIDs, metas
    elif return_meta:
        return imgs, metas
    elif returnIDs:
        return imgs, sIDs
    else:
        return imgs

def ReadSeriesV2(folder_path, returnIDs=False, return_meta=False, series_ids=None, series2array=True, taginits2ignore=[]):
    #This function is identical to ReadSeries, but it uses pydicom instead of SimpleITK. But they do not return the same results (TODO!)
    if not bool(series_ids):
        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(folder_path)
    if not isinstance(series_ids, list):    
        series_ids = [series_ids]
    imgs = []
    metas = []
    sIDs = []
    for sid in series_ids:
        #read the dicom series

        with contextlib.suppress(Exception):
            dicom_names = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.dcm') or f.endswith('.IMA')]
            dicom_names = sorted(dicom_names, key=lambda x: pydicom.dcmread(x).AcquisitionTime)
            image = []
            meta = []
            for d in dicom_names:
                ds = pydicom.dcmread(d)
                # check if the sid matches with the one provided
                if ds.SeriesInstanceUID == sid:
                    if return_meta:
                        headers = {elem: getattr(ds, elem) for elem in ds.dir() if elem != "PixelData"}
                        meta.append(headers)
                    image.append(ds.pixel_array)
            imgs.append(np.stack(image))
            sIDs.append(sid)
            if return_meta:
                metas.append(meta[0]) #TODO: make is possible to choose the slice to return the meta data, currently only the first one

    if series2array:
        imgs = np.array(imgs)
    if return_meta and returnIDs:
        return imgs, sIDs, metas
    elif return_meta:
        return imgs, metas
    elif returnIDs:
        return imgs, sIDs
    else:
        return imgs

def ReadDICOMDIR(dicomdir_path, returnIDs=False):
    ds = pydicom.dcmread(dicomdir_path)
    imgs = []
    sIDs = []

    # Iterate through the PATIENT records
    for patient in ds.patient_records:
        pID = f"{patient.PatientID}_{patient.PatientName}"
        # Find all the STUDY records for the patient
        studies = [
            ii for ii in patient.children if ii.DirectoryRecordType == "STUDY"
        ]
        for study in studies:
            sID = f"{study.StudyID}"
            # Find all the SERIES records in the study
            all_series = [
                ii for ii in study.children if ii.DirectoryRecordType == "SERIES"
            ]
            for series in all_series:
                # Find all the IMAGE records in the series
                images = [
                    ii for ii in series.children
                    if ii.DirectoryRecordType == "IMAGE"
                ]

                # Get the absolute file path to each instance                
                elems = [ii["ReferencedFileID"] for ii in images] # Each IMAGE contains a relative file path to the root directory
                paths = [[ee.value] if ee.VM == 1 else ee.value for ee in elems] # Make sure the relative file path is always a list of str
                paths = [os.path.join(os.path.dirname(dicomdir_path), os.sep.join(p)) for p in paths]

                with contextlib.suppress(Exception):
                    reader = sitk.ImageSeriesReader()
                    reader.SetFileNames(paths)
                    image = reader.Execute()
                    imgs.append(sitk.GetArrayFromImage(image))
                    sIDs.append(f"{pID}_{sID}_{series.SeriesInstanceUID}")
    imgs = np.array(imgs)
    return (imgs, sIDs) if returnIDs else imgs

def FileRead(file_path, return_data=True, return_ds=False):
    ds = pydicom.dcmread(file_path)
    if return_data and not return_ds:
        return ds.pixel_array
    elif not return_data and return_ds:
        return ds
    else:
        return ds.pixel_array, ds

def toNIFTI(dicom_path, nifti_path, isSeries=True, nifti_file_name=None):
    if isSeries:
        imgs, IDs = ReadSeries(dicom_path, returnIDs=True)
    else:
        imgs, IDs = ReadDICOMDIR(dicom_path, returnIDs=True)
    for i, (im, ID) in enumerate(zip(imgs, IDs)):
        if nifti_file_name is not None:
            ID = f"{nifti_file_name}_{str(i)}"
        stikSave(im, f"{nifti_path}/{ID}.nii.gz")