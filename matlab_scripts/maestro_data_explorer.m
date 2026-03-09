function maestro_data_explorer()
    % MAESTRO_DATA_EXPLORER - A simple script to explore missing neurons in Maestro data.
    parentDir = sprintf('%s/data/fiona_sst', fileparts(pwd));
    count_trials_in_all_sessions(parentDir);
end

function [total_trials, errors] = count_trials_in_all_sessions(parentDir)
    disp(['Exploring parent directory: ', parentDir]);
    fiDirs = get_parent_dir_valid_subdirs(parentDir);
    N = numel(fiDirs);
    disp(['Found ', num2str(N), ' valid session directories.']);

    total_trials = 0;
    errors = {};

    for i = 1:N
        sessionDirPath = fullfile(parentDir, fiDirs{i});
        try
            trials_num = count_trials_in_single_session_dir(sessionDirPath);
            total_trials = total_trials + trials_num;
            fprintf('Session %s: %d trial(s) found.\n', fiDirs{i}, trials_num);
        catch ME
            fprintf('Error exploring session %s: %s\n', fiDirs{i}, ME.message);
            errors{end+1} = sprintf('Session %s: %s', fiDirs{i}, ME.message); %#ok<AGROW>
        end
    end

    fprintf('Total trials across all sessions: %d\n', total_trials);
    if ~isempty(errors)
        fprintf('Errors encountered in the following sessions:\n');
        for j = 1:numel(errors)
            fprintf('%s\n', errors{j});
        end
    end
end

function trials_num = count_trials_in_single_session_dir(session_dir_path)

    folderPath = session_dir_path; %sprintf('%s/data/fiona_sst/fi211109', fileparts(pwd));
    % disp(['Exploring session folder: ', folderPath]);
    
    candidates = dir(folderPath);
    candidates = candidates(~[candidates.isdir]);

    isTrial = false(numel(candidates),1);
    for i = 1:numel(candidates)
        [~,~,ext] = fileparts(candidates(i).name);
        extLower = lower(ext);
        if ~isempty(regexp(extLower, '^\.\d{3,4}$', 'once')) || ...
        ~isempty(regexp(extLower, '^\.0+\d+$', 'once'))
            isTrial(i) = true;
        end
    end

    files = candidates(isTrial);
    if isempty(files)
        patt = {'*.000*','*.001*','*.002*','*.003*'};
        files = [];
        for k = 1:numel(patt)
            files = [files; dir(fullfile(folderPath, patt{k}))]; %#ok<AGROW>
        end
    end
    if isempty(files)
        % safeClose(fidLog);
        error('No trial-like files found in %s. Adjust detection patterns above if needed.', folderPath);
    end

    [~,ord] = sort({files.name});
    files   = files(ord);
    % fprintf('Found %d trial file(s).\n', numel(files));
    trials_num = numel(files);
    % fprintf('%s\n', files(1).name);
    % data = readcxdata(fullfile(folderPath, files(1).name));
    % disp(length(data.sortedSpikes));
end

function fiDirs = get_parent_dir_valid_subdirs(parentDir)
    % Get only directories (exclude '.' and '..')
    D = dir(parentDir);
    isDir = [D.isdir] & ~ismember({D.name},{'.','..'});

    % Match names like fi21XXXX (XXXX = 4 digits)
    names = {D(isDir).name};
    mask  = ~cellfun('isempty', regexp(names, '^fi21\d{4}$', 'once'));

    fiDirs = names(mask);
    if isempty(fiDirs)
        error('No folders matching "fi21XXXX" found in %s.', parentDir);
    end

    % Sort (lexical is fine since digits are fixed-width)
    fiDirs = sort(fiDirs);
end