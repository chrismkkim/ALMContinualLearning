% fileName = "BAYLORJH035_FOV1_2021_12_09_2022_02_15.mat";
% load("/Users/kimchm/Documents/KimNiNature2024/MatFilesFromJaeHyun/source/BAYLORJH035_FOV1_2021_12_09_2022_02_15.mat")
% load("/Users/kimchm/Documents/KimNiNature2024/MatFilesFromJaeHyun/source/BAYLORJH035_FOV1_2022_02_24_2022_05_23.mat")

saveCDdotproduct = 'CDdotproduct_matlab.txt';
f = fopen(saveCDdotproduct,'w');

folderPath = "/Users/kimchm/Documents/KimNiNature2024/MatFilesFromJaeHyun/source/";
filePattern = fullfile(folderPath, 'BAYLOR*.mat'); 
dirContents = dir(filePattern);

for i = 1:length(dirContents)

    data = load([dirContents(i).folder, '/', dirContents(i).name]);

    % Time and Indexing Parameters
    tsample   = 1.57;
    tdelay    = 2.87;
    tresponse = 4.17;
    dt        = 1/6;
    ntimestep = 47;    
    tvec      = dt * (0:ntimestep-1);
    
    % find the time indices of the onset of delay period and response
    tix_delay    = find(tvec > tdelay, 1, 'first');
    tix_response = find(tvec > tresponse, 1, 'first');
    
    % opto data
    opto_sess1 = data.deconvolved{1,1};
    opto_sess2 = data.deconvolved{2,1};
    
    % use the correct trials of lick right and lick left
    [ncell, ncat] = size(opto_sess1);
    lick_rightcorrect = 1; 
    lick_leftcorrect  = 2;

    % number of trials for each type
    ntrials_1R = size(opto_sess1{1, lick_rightcorrect}, 1);
    ntrials_1L = size(opto_sess1{1, lick_leftcorrect}, 1);
    ntrials_2R = size(opto_sess2{1, lick_rightcorrect}, 1);
    ntrials_2L = size(opto_sess2{1, lick_leftcorrect}, 1);
    
    ntrials_train_1R = floor(ntrials_1R/2);
    ntrials_train_1L = floor(ntrials_1L/2);
    ntrials_train_2R = floor(ntrials_2R/2);
    ntrials_train_2L = floor(ntrials_2L/2);

    % Preallocate matrices 
    opto_sess1_lickright = zeros(ncell, ntimestep);
    opto_sess1_lickleft  = zeros(ncell, ntimestep);
    opto_sess2_lickright = zeros(ncell, ntimestep);
    opto_sess2_lickleft  = zeros(ncell, ntimestep);
    
    % Loop through cells to average across trials
    for cell_idx = 1:ncell
        opto_sess1_lickright(cell_idx, :) = mean(opto_sess1{cell_idx, lick_rightcorrect}(1:ntrials_train_1R,:), 1);
        opto_sess1_lickleft(cell_idx, :)  = mean(opto_sess1{cell_idx, lick_leftcorrect}(1:ntrials_train_1L,:), 1);
        opto_sess2_lickright(cell_idx, :) = mean(opto_sess2{cell_idx, lick_rightcorrect}(1:ntrials_train_2R,:), 1);
        opto_sess2_lickleft(cell_idx, :)  = mean(opto_sess2{cell_idx, lick_leftcorrect}(1:ntrials_train_2L,:), 1);
    end
    
    % Compute coding direction 
    CD_sess1 = opto_sess1_lickright - opto_sess1_lickleft;
    CD_sess2 = opto_sess2_lickright - opto_sess2_lickleft;
    
    % Calculate mean across the delay period
    CD_sess1_population = mean(CD_sess1(:, tix_delay:tix_response-1), 2);
    CD_sess2_population = mean(CD_sess2(:, tix_delay:tix_response-1), 2);
    
    % Dot product
    CD_dotproduct = dot(CD_sess1_population, CD_sess2_population) / ...
                    (norm(CD_sess1_population) * norm(CD_sess2_population));
    
    % print output
    char_fname = char(dirContents(i).name);
    fprintf('%s, CD dot product: %f\n', char_fname(1:end-4), CD_dotproduct);    
    fprintf(f, '%s, %f\n', char_fname(1:end-4), CD_dotproduct);

end

fclose(f);

% 
% % sort the text file
% % 1. Open and read the file manually
% fID = fopen(saveCDdotproduct, 'r');
% data = textscan(fID, '%s %f', 'Delimiter', ','); % Read columns as string and float
% fclose(fID);                 % Close manually as requested
% 
% % 2. Extract columns
% sessionIDs = data{1};        % This is a cell array of strings
% metrics = data{2};           % This is a numerical array
% 
% % 3. Sort by the first column (sessionIDs)
% % sort() returns the sorted values and the original indices 'idx'
% [sortedIDs, idx] = sort(sessionIDs);
% sortedMetrics = metrics(idx); % Reorder numerical values to match
% 
% % 4. Open and write the sorted data back to the file
% saveCDdotproduct_sorted = 'MatCDdotproduct_sorted.txt';
% fID_out = fopen(saveCDdotproduct_sorted, 'w');
% for i = 1:length(sortedIDs)
%     % Write the string and its corresponding number
%     fprintf(fID_out, '%s,%.4f\n', sortedIDs{i}, sortedMetrics(i));
% end
% fclose(fID_out);