addpath(genpath('/Users/kimchm/Documents/GitHub/matnwb'));
generateCore()

% Load the NWB file
% version = "version 0.240912.1925";
version = "version 0.251008.1146";
% version = "version 0.251010.2006";
% version = "version 001188.draft";
animal = "sub-BAYLORJH035/";
sess = "sub-BAYLORJH035_ses-FOV3-2022-01-12_ophys.nwb";
dirpath = "/Users/kimchm/Documents/KimNiNature2024/" + version + "/";
filepath = dirpath + animal + sess;
nwbFile = nwbRead(filepath);