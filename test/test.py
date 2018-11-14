import pydicom
import os
import numpy as np

class DicomFile:
    def __init__(self, file_path, dataset):
        self.file_path = file_path
        self.dataset = dataset

filepath = '/DB/rhome/qyzheng/Desktop/Link to renji_data/bladder/2013-2015/D0501566/HKSY1EDT/DHLEKEFK/I1000000'
df = DicomFile(filepath, pydicom.read_file(filepath))
print df.dataset.AccessionNumber
print df.dataset.SeriesDescription
print df.dataset.SeriesInstanceUID
