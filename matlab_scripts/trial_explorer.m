function trial_explorer()
    file_path = sprintf('%s/data/fiona_sst/fi210713/fi210713a.0193', fileparts(pwd));
    d = readcxdata(file_path);
    disp(d)
end