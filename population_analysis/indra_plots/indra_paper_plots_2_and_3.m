%% indra_paper_plots_2_and_3.m
% Translated from indra_paper_plots_2_and_3.ipynb
% Reproduces Figures 2 and 3 from Indra et al. analysis
%
% Requirements:
%   - cell trial data saved as .mat (converted from pickle)
%   - SSRT Excel file
%   - Statistics and Machine Learning Toolbox (for signrank, ranksum)
%   - Parallel Computing Toolbox (optional, for parfor)

clear; clc; close all;

%% ========================================================================
%  1. Load data and preprocess
%  1. Load cell metadata
%  2. Leave only MSN cells
%  3. Load the SSRT DataFrame
%  ========================================================================

base_path = fullfile(fileparts(fileparts(pwd)), 'data', 'unified_cell_trial_data');

% --- Load MSN cell database ---
% NOTE: Python pickle files cannot be loaded directly in MATLAB.
% You must first export the pickle to .mat in Python using the conversion
% function provided in the notebook (see cell after loading cell_df).
%
% The .mat file contains each DataFrame column as a separate variable.
% Fields include: cell_ID, cell_type, session, type, dir, trial_failed,
%   reaction_time, ssd_number, ssd_len, grade, go_cue, stop_cue,
%   first_relevant_saccade, neural_data, trial_number, trial_session, is_slow_go

pickle_file = fullfile(base_path, 'alt_saccade_params_msn_fiona_cell_trial_data.mat');
fprintf('Loading cell data from: %s\n', pickle_file);
loaded = load(pickle_file);

% Each column is saved as a separate field
field_names = fieldnames(loaded);
fprintf('Loaded %d fields\n', numel(field_names));

% Get number of rows from first field
first_field = loaded.(field_names{1});
if iscell(first_field)
    n_rows = numel(first_field);
elseif isvector(first_field)
    n_rows = length(first_field);
else
    n_rows = size(first_field, 1);
end
fprintf('Number of rows: %d\n', n_rows);

% Build table manually - each field should become a column
table_data = struct();
for i = 1:numel(field_names)
    fname = field_names{i};
    field_data = loaded.(fname);
    
    % Ensure column vector
    if isvector(field_data) && size(field_data, 2) > 1
        field_data = field_data(:);
    end
    
    table_data.(fname) = field_data;
end

cell_df = struct2table(table_data);
fprintf('Table created with %d rows x %d columns\n', height(cell_df), width(cell_df));

% Convert cell arrays of strings to categorical/string
if iscell(cell_df.cell_type)
    cell_type_str = cell(size(cell_df.cell_type));
    for i = 1:numel(cell_df.cell_type)
        val = cell_df.cell_type{i};
        cell_type_str{i} = char(val);
    end
    cell_df.cell_type = categorical(cell_type_str);
end

if iscell(cell_df.type)
    type_str = cell(size(cell_df.type));
    for i = 1:numel(cell_df.type)
        val = cell_df.type{i};
        type_str{i} = char(val);
    end
    cell_df.type = categorical(type_str);
end

if iscell(cell_df.session)
    session_str = cell(size(cell_df.session));
    for i = 1:numel(cell_df.session)
        val = cell_df.session{i};
        session_str{i} = char(val);
    end
    cell_df.session = string(session_str);
end

% Filter to MSN cells only
fprintf('Filtering to MSN cells...\n');
msn_mask = (cell_df.cell_type == 'msn');
cell_df = cell_df(msn_mask, :);
fprintf('Filtered to %d MSN cells (from %d total)\n', height(cell_df), n_rows);

%% Load SSRT data
ssrt_file = fullfile(fileparts(base_path), 'SSRT_F.xlsx');
ssrt_db = readtable(ssrt_file);
ssrt_db = ssrt_db(:, {'session', 'Left', 'Right'});

% Compute max of Left and Right
ssrt_db.max_ssrt = max(ssrt_db.Left, ssrt_db.Right);
% Fill NaN max values with the overall max
ssrt_db.max_ssrt(isnan(ssrt_db.max_ssrt)) = max(ssrt_db.max_ssrt);

% Fix session names (remove apostrophes) and convert to string
ssrt_db.session = string(ssrt_db.session);
ssrt_db.session = strrep(ssrt_db.session, '''', '');
ssrt_db = sortrows(ssrt_db, 'session');

%% Compute mean SSD per session (for ssd_number < 5)
valid_ssd = cell_df(cell_df.ssd_number < 5, :);
[sessions_u, ~, session_idx] = unique(valid_ssd.session);
mean_ssd_per_session = zeros(numel(sessions_u), 1);
for i = 1:numel(sessions_u)
    mean_ssd_per_session(i) = round(nanmean(valid_ssd.ssd_len(session_idx == i)));
end
ssd_per_session = table(sessions_u, mean_ssd_per_session, ...
    'VariableNames', {'session', 'mean_ssd_len'});

%% Merge SSRT and SSD tables
ssrt_ssd_df = innerjoin(ssrt_db, ssd_per_session, 'Keys', 'session');

% Fill NaN Right SSRT with mean
ssrt_ssd_df.Right(isnan(ssrt_ssd_df.Right)) = nanmean(ssrt_ssd_df.Right);

% Compute slow-go threshold
ssrt_ssd_df.slow_go_thresh = round(ssrt_ssd_df.Right + ssrt_ssd_df.mean_ssd_len);

%% Count MSN cells
unique_cells = unique(cell_df(cell_df.cell_type == 'msn', :).cell_ID);
num_of_msn_cells = numel(unique_cells);
fprintf('Number of MSN cells: %d\n', num_of_msn_cells);

%% Merge slow_go_thresh into cell_df and mark slow GO trials
% Create a lookup map: session -> slow_go_thresh
thresh_map = containers.Map();
for i = 1:height(ssrt_ssd_df)
    sess = char(ssrt_ssd_df.session(i));
    thresh_map(sess) = ssrt_ssd_df.slow_go_thresh(i);
end
fprintf('Created threshold map for %d sessions\n', length(thresh_map.keys));

slow_go_thresh_out = NaN(height(cell_df), 1);

% Pre-extract session data for parfor loop
sessions_char = cell(height(cell_df), 1);
for i = 1:height(cell_df)
    sessions_char{i} = char(cell_df.session(i));
end

parfor i = 1:height(cell_df)
    sess = sessions_char{i};
    if thresh_map.isKey(sess)
        slow_go_thresh_out(i) = thresh_map(sess);
    end
end

cell_df.slow_go_thresh = slow_go_thresh_out;
cell_df.is_slow_go = NaN(height(cell_df), 1);

fprintf('Merged slow_go_thresh into cell_df\n');

% Mark slow GO trials
go_mask = cell_df.type == 'GO';
cell_df.is_slow_go(go_mask) = ...
    ~isnan(cell_df.slow_go_thresh(go_mask)) & ...
    (cell_df.reaction_time(go_mask) >= cell_df.slow_go_thresh(go_mask));

% Mark slow CONT trials (successful only)
cont_mask = (cell_df.type == 'CONT') & (cell_df.trial_failed == false);
cell_df.is_slow_go(cont_mask) = ...
    ~isnan(cell_df.slow_go_thresh(cont_mask)) & ...
    (cell_df.reaction_time(cont_mask) >= cell_df.slow_go_thresh(cont_mask));

fprintf('Total trials after filtering: %d\n', height(cell_df));
fprintf('is_slow_go counts: True=%d, False=%d, NaN=%d\n', ...
    sum(cell_df.is_slow_go == 1), sum(cell_df.is_slow_go == 0), ...
    sum(isnan(cell_df.is_slow_go)));

%% ========================================================================
%  Configuration
%  ========================================================================

% From Indra et al., 2020, methods section page 5;380:
% The recording cylinder was implanted on the left hemisphere for monkey F(iona)
CONTRA_DIR = 0;  % Contra direction for monkey Fiona
SMOOTH_SIGMA = 20;
BIN_SIZE = 1;

%% ========================================================================
%  2. First condition filtering (Criterion #1)
%  A significant firing rate increase around contralateral saccade onset
%  (-100 ms to +100 ms from saccade onset versus baseline from -300 ms to go)
%  on the go trials (one-tailed Wilcoxon signed-rank, P <= 0.05)
%  ========================================================================

unique_cell_ids = unique(cell_df.cell_ID);
n_cells = numel(unique_cell_ids);
fprintf('Processing %d cells for Condition #1...\n', n_cells);

cond1_stats = NaN(n_cells, 1);
cond1_pvals = NaN(n_cells, 1);
cond1_errors = cell(n_cells, 1);

parfor ci = 1:n_cells
    cid = unique_cell_ids(ci);
    try
        cell_data = cell_df(cell_df.cell_ID == cid, :);
        [baseline_rates, saccade_rates] = get_cell_spike_counts_condition_1( ...
            cell_data, CONTRA_DIR);
        
        if numel(baseline_rates) < 5
            cond1_pvals(ci) = NaN;
            continue;
        end
        
        % One-tailed Wilcoxon signed-rank test: baseline < saccade
        [p, ~, stats] = signrank(baseline_rates, saccade_rates, 'Tail', 'left');
        cond1_pvals(ci) = p;
        if isstruct(stats) && isfield(stats, 'signedrank')
            cond1_stats(ci) = stats.signedrank;
        end
    catch ME
        cond1_errors{ci} = ME.message;
    end
end

% Apply Holm-Bonferroni correction
valid_mask = ~isnan(cond1_pvals);
[cond1_reject, cond1_pvals_corrected] = holm_bonferroni(cond1_pvals(valid_mask), 0.05);

% Use uncorrected p < 0.05 (as in notebook)
cond1_regular_reject = cond1_pvals < 0.05;
significant_cell_ids_cond1 = unique_cell_ids(cond1_regular_reject);
fprintf('Condition #1 significant cells (uncorrected p<0.05): %d\n', ...
    numel(significant_cell_ids_cond1));

%% ========================================================================
%  3. Second condition filtering (Criterion #2)
%  A significant decrease in correct stop trials vs. slow go in the
%  contralateral direction (0-400 ms bin from stop signal onset)
%  (one-tailed Mann-Whitney U / Wilcoxon rank-sum, P <= 0.05)
%  ========================================================================

fprintf('Processing %d cells for Condition #2...\n', n_cells);

cond2_stats = NaN(n_cells, 1);
cond2_pvals = NaN(n_cells, 1);
cond2_errors = cell(n_cells, 1);

parfor ci = 1:n_cells
    cid = unique_cell_ids(ci);
    try
        cell_data = cell_df(cell_df.cell_ID == cid, :);
        [slow_go_rates, correct_stop_rates] = get_cell_spike_counts_condition_2( ...
            cell_data, CONTRA_DIR);
        
        if numel(slow_go_rates) < 5 || numel(correct_stop_rates) < 5
            cond2_pvals(ci) = NaN;
            continue;
        end
        
        % One-tailed rank-sum test: correct_stop < slow_go (decrease in stop)
        [p, ~, stats] = ranksum(correct_stop_rates, slow_go_rates, 'Tail', 'left');
        cond2_pvals(ci) = p;
        if isstruct(stats) && isfield(stats, 'ranksum')
            cond2_stats(ci) = stats.ranksum;
        end
    catch ME
        cond2_errors{ci} = ME.message;
    end
end

% Apply Holm-Bonferroni correction
valid_mask2 = ~isnan(cond2_pvals);
[cond2_reject, cond2_pvals_corrected] = holm_bonferroni(cond2_pvals(valid_mask2), 0.05);

% Use uncorrected p < 0.05 (as in notebook)
cond2_regular_reject = cond2_pvals < 0.05;
significant_cell_ids_cond2 = unique_cell_ids(cond2_regular_reject);
fprintf('Condition #2 significant cells (uncorrected p<0.05): %d\n', ...
    numel(significant_cell_ids_cond2));

%% ========================================================================
%  4. Define GO neurons (intersection of conditions 1 and 2)
%  ========================================================================

go_neurons = intersect(significant_cell_ids_cond1, significant_cell_ids_cond2);
fprintf('Number of GO neurons (intersection): %d\n', numel(go_neurons));

%% ========================================================================
%  5. Compute population PSTH for Figure 3 (saccade-aligned)
%     Trial types: GO, STOP (error), CONT — saccade-aligned
%  ========================================================================

fprintf('Computing population PSTHs for Figure 3...\n');

saccade_epok = [-300, 300];
baseline_epok = [-400, 0];

n_go = numel(go_neurons);

% Pre-allocate storage for all neurons' PSTHs
% First, compute one cell to get array sizes
sample_data = cell_df(cell_df.cell_ID == go_neurons(1), :);
[sample_bins, ~, ~] = calculate_psth_for_cell(sample_data, ...
    'first_relevant_saccade', saccade_epok, BIN_SIZE, SMOOTH_SIGMA, ...
    true, false, CONTRA_DIR, 'GO');
n_saccade_bins = numel(sample_bins);

[sample_bins_bl, ~, ~] = calculate_psth_for_cell(sample_data, ...
    'go_cue', baseline_epok, BIN_SIZE, SMOOTH_SIGMA, ...
    true, false, CONTRA_DIR, 'GO');
n_baseline_bins = numel(sample_bins_bl);

% Storage arrays
y_saccade_GO = NaN(n_go, n_saccade_bins);
y_saccade_STOP = NaN(n_go, n_saccade_bins);
y_saccade_CONT = NaN(n_go, n_saccade_bins);

y_baseline_GO = NaN(n_go, n_baseline_bins);
y_baseline_STOP = NaN(n_go, n_baseline_bins);
y_baseline_CONT = NaN(n_go, n_baseline_bins);

parfor ni = 1:n_go
    cid = go_neurons(ni);
    cdata = cell_df(cell_df.cell_ID == cid, :);
    
    % Saccade-aligned PSTHs
    [~, fr_go, ~] = calculate_psth_for_cell(cdata, ...
        'first_relevant_saccade', saccade_epok, BIN_SIZE, SMOOTH_SIGMA, ...
        true, false, CONTRA_DIR, 'GO');
    [~, fr_stop, ~] = calculate_psth_for_cell(cdata, ...
        'first_relevant_saccade', saccade_epok, BIN_SIZE, SMOOTH_SIGMA, ...
        false, true, CONTRA_DIR, 'STOP');
    [~, fr_cont, ~] = calculate_psth_for_cell(cdata, ...
        'first_relevant_saccade', saccade_epok, BIN_SIZE, SMOOTH_SIGMA, ...
        true, false, CONTRA_DIR, 'CONT');
    
    y_saccade_GO(ni, :) = fr_go;
    y_saccade_STOP(ni, :) = fr_stop;
    y_saccade_CONT(ni, :) = fr_cont;
    
    % Baseline-aligned PSTHs
    [~, fr_go_bl, ~] = calculate_psth_for_cell(cdata, ...
        'go_cue', baseline_epok, BIN_SIZE, SMOOTH_SIGMA, ...
        true, false, CONTRA_DIR, 'GO');
    [~, fr_stop_bl, ~] = calculate_psth_for_cell(cdata, ...
        'go_cue', baseline_epok, BIN_SIZE, SMOOTH_SIGMA, ...
        false, true, CONTRA_DIR, 'STOP');
    [~, fr_cont_bl, ~] = calculate_psth_for_cell(cdata, ...
        'go_cue', baseline_epok, BIN_SIZE, SMOOTH_SIGMA, ...
        true, false, CONTRA_DIR, 'CONT');
    
    y_baseline_GO(ni, :) = fr_go_bl;
    y_baseline_STOP(ni, :) = fr_stop_bl;
    y_baseline_CONT(ni, :) = fr_cont_bl;
end

%% Plot saccade-aligned population PSTH (Figure 3a)
x_saccade = sample_bins;

mean_y_saccade = nanmean(y_saccade_GO, 1);
sem_y_saccade  = nanstd(y_saccade_GO, 0, 1) / sqrt(size(y_saccade_GO, 1));

mean_y_saccade_stop = nanmean(y_saccade_STOP, 1);
sem_y_saccade_stop  = nanstd(y_saccade_STOP, 0, 1) / sqrt(size(y_saccade_STOP, 1));

mean_y_saccade_cont = nanmean(y_saccade_CONT, 1);
sem_y_saccade_cont  = nanstd(y_saccade_CONT, 0, 1) / sqrt(size(y_saccade_CONT, 1));

figure('Position', [100 100 800 500]);
hold on;

% GO trials (green)
fill([x_saccade, fliplr(x_saccade)], ...
     [mean_y_saccade - sem_y_saccade, fliplr(mean_y_saccade + sem_y_saccade)], ...
     [0.56 0.93 0.56], 'FaceAlpha', 0.3, 'EdgeColor', 'none');
plot(x_saccade, mean_y_saccade, 'Color', [0 0.5 0], 'LineWidth', 2, 'DisplayName', 'GO');

% STOP trials (black)
fill([x_saccade, fliplr(x_saccade)], ...
     [mean_y_saccade_stop - sem_y_saccade_stop, fliplr(mean_y_saccade_stop + sem_y_saccade_stop)], ...
     [0.7 0.7 0.7], 'FaceAlpha', 0.3, 'EdgeColor', 'none');
plot(x_saccade, mean_y_saccade_stop, 'k', 'LineWidth', 2, 'DisplayName', 'ERROR STOP');

% CONT trials (blue)
fill([x_saccade, fliplr(x_saccade)], ...
     [mean_y_saccade_cont - sem_y_saccade_cont, fliplr(mean_y_saccade_cont + sem_y_saccade_cont)], ...
     [0.68 0.85 0.90], 'FaceAlpha', 0.3, 'EdgeColor', 'none');
plot(x_saccade, mean_y_saccade_cont, 'b', 'LineWidth', 2, 'DisplayName', 'CONT');

xline(0, 'r--', 'Alpha', 0.5, 'DisplayName', 'Saccade onset');
xlabel('Time from saccade onset (ms)', 'FontSize', 12);
ylabel('Firing rate (spikes/s)', 'FontSize', 12);
title(sprintf('Population PSTH - Saccade-aligned\n%d/%d MSN cells', n_go, num_of_msn_cells), 'FontSize', 13);
legend('Location', 'best');
grid on; set(gca, 'GridAlpha', 0.3);
hold off;

%% Plot baseline-aligned population PSTH (Figure 3b)
x_baseline = sample_bins_bl;

mean_y_baseline = nanmean(y_baseline_GO, 1);
sem_y_baseline  = nanstd(y_baseline_GO, 0, 1) / sqrt(size(y_baseline_GO, 1));

mean_y_baseline_stop = nanmean(y_baseline_STOP, 1);
sem_y_baseline_stop  = nanstd(y_baseline_STOP, 0, 1) / sqrt(size(y_baseline_STOP, 1));

mean_y_baseline_cont = nanmean(y_baseline_CONT, 1);
sem_y_baseline_cont  = nanstd(y_baseline_CONT, 0, 1) / sqrt(size(y_baseline_CONT, 1));

figure('Position', [100 100 800 500]);
hold on;

% GO trials (green)
fill([x_baseline, fliplr(x_baseline)], ...
     [mean_y_baseline - sem_y_baseline, fliplr(mean_y_baseline + sem_y_baseline)], ...
     [0.56 0.93 0.56], 'FaceAlpha', 0.3, 'EdgeColor', 'none');
plot(x_baseline, mean_y_baseline, 'Color', [0 0.5 0], 'LineWidth', 2, 'DisplayName', 'GO');

% STOP trials (black)
fill([x_baseline, fliplr(x_baseline)], ...
     [mean_y_baseline_stop - sem_y_baseline_stop, fliplr(mean_y_baseline_stop + sem_y_baseline_stop)], ...
     [0.7 0.7 0.7], 'FaceAlpha', 0.3, 'EdgeColor', 'none');
plot(x_baseline, mean_y_baseline_stop, 'k', 'LineWidth', 2, 'DisplayName', 'ERROR STOP');

% CONT trials (blue)
fill([x_baseline, fliplr(x_baseline)], ...
     [mean_y_baseline_cont - sem_y_baseline_cont, fliplr(mean_y_baseline_cont + sem_y_baseline_cont)], ...
     [0.68 0.85 0.90], 'FaceAlpha', 0.3, 'EdgeColor', 'none');
plot(x_baseline, mean_y_baseline_cont, 'b', 'LineWidth', 2, 'DisplayName', 'CONT');

xline(0, 'r--', 'Alpha', 0.5, 'DisplayName', 'Go cue');
xlabel('Time from go cue (ms)', 'FontSize', 12);
ylabel('Firing rate (spikes/s)', 'FontSize', 12);
title(sprintf('Population PSTH - Baseline-aligned\n%d/%d MSN cells', n_go, num_of_msn_cells), 'FontSize', 13);
legend('Location', 'best');
grid on; set(gca, 'GridAlpha', 0.3);
hold off;

%% Strip plot: baseline vs saccade mean firing rates per neuron
figure('Position', [100 100 600 400]);
hold on;

baseline_means_list = NaN(n_go, 1);
saccade_means_list = NaN(n_go, 1);

for ni = 1:n_go
    bl_mean = nanmean(y_baseline_GO(ni, :));
    bl_sem  = nanstd(y_baseline_GO(ni, :), 0) / sqrt(sum(~isnan(y_baseline_GO(ni, :))));
    sc_mean = nanmean(y_saccade_GO(ni, :));
    sc_sem  = nanstd(y_saccade_GO(ni, :), 0) / sqrt(sum(~isnan(y_saccade_GO(ni, :))));
    
    baseline_means_list(ni) = bl_mean;
    saccade_means_list(ni) = sc_mean;
    
    errorbar(0, bl_mean, bl_sem, 'o', 'Color', [0.6 0.6 0.6], ...
        'MarkerSize', 4, 'CapSize', 2, 'LineWidth', 1, 'MarkerFaceColor', [0.6 0.6 0.6]);
    errorbar(1, sc_mean, sc_sem, 'o', 'Color', [0.6 0.6 0.6], ...
        'MarkerSize', 4, 'CapSize', 2, 'LineWidth', 1, 'MarkerFaceColor', [0.6 0.6 0.6]);
    plot([0, 1], [bl_mean, sc_mean], '-', 'Color', [0.6 0.6 0.6 0.4], 'LineWidth', 1);
end

% Population mean +/- SEM
pop_bl_mean = mean(baseline_means_list);
pop_bl_sem  = std(baseline_means_list) / sqrt(numel(baseline_means_list));
pop_sc_mean = mean(saccade_means_list);
pop_sc_sem  = std(saccade_means_list) / sqrt(numel(saccade_means_list));

errorbar(0, pop_bl_mean, pop_bl_sem, 'o', 'Color', [0 0.5 0], ...
    'MarkerSize', 10, 'LineWidth', 2, 'CapSize', 5, 'MarkerFaceColor', [0 0.5 0]);
errorbar(1, pop_sc_mean, pop_sc_sem, 'o', 'Color', [0 0.5 0], ...
    'MarkerSize', 10, 'LineWidth', 2, 'CapSize', 5, 'MarkerFaceColor', [0 0.5 0]);
plot([0, 1], [pop_bl_mean, pop_sc_mean], '-', 'Color', [0 0.5 0], 'LineWidth', 2);

xlim([-0.3, 1.3]);
xticks([0, 1]);
xticklabels({'Baseline\n(-400 to 0 ms)', 'Saccade\n(-100 to +100 ms)'});
ylabel('Mean PSTH (spikes/s)', 'FontSize', 12);
title(sprintf('GO Neuron Activity: Baseline vs Saccade-aligned\n(n=%d neurons)', n_go), 'FontSize', 13);
grid on; set(gca, 'GridAlpha', 0.3, 'YGrid', 'on', 'XGrid', 'off');
hold off;

%% ========================================================================
%  6. Figure 2: Go-cue and Stop-signal aligned PSTHs
%     (Using only slow-GO trials)
%  ========================================================================

fprintf('Computing population PSTHs for Figure 2...\n');

go_cue_epok = [0, 600];
stop_sig_epok = [-300, 600];

% Get bin sizes
sample_data_f2 = cell_df(cell_df.cell_ID == go_neurons(1) & ...
    (cell_df.is_slow_go == 1 | isnan(cell_df.is_slow_go)), :);

[sample_bins_gc, ~, ~] = calculate_psth_for_cell(sample_data_f2, ...
    'go_cue', go_cue_epok, BIN_SIZE, SMOOTH_SIGMA, ...
    true, false, CONTRA_DIR, 'GO');
n_gc_bins = numel(sample_bins_gc);

[sample_bins_ss, ~, ~] = calculate_psth_for_cell(sample_data_f2, ...
    'stop_cue', stop_sig_epok, BIN_SIZE, SMOOTH_SIGMA, ...
    true, false, CONTRA_DIR, 'GO');
n_ss_bins = numel(sample_bins_ss);

% Storage for Figure 2
y_go_cue_GO   = NaN(n_go, n_gc_bins);
y_go_cue_STOP = NaN(n_go, n_gc_bins);
y_go_cue_CONT = NaN(n_go, n_gc_bins);

y_stop_sig_GO   = NaN(n_go, n_ss_bins);
y_stop_sig_STOP = NaN(n_go, n_ss_bins);
y_stop_sig_CONT = NaN(n_go, n_ss_bins);

parfor ni = 1:n_go
    cid = go_neurons(ni);
    % Filter to slow GO trials + non-GO (as in notebook)
    cdata = cell_df(cell_df.cell_ID == cid & ...
        (cell_df.is_slow_go == 1 | isnan(cell_df.is_slow_go)), :);
    
    % Go-cue aligned
    [~, fr, ~] = calculate_psth_for_cell(cdata, 'go_cue', go_cue_epok, ...
        BIN_SIZE, SMOOTH_SIGMA, true, false, CONTRA_DIR, 'GO');
    y_go_cue_GO(ni, :) = fr;
    
    [~, fr, ~] = calculate_psth_for_cell(cdata, 'go_cue', go_cue_epok, ...
        BIN_SIZE, SMOOTH_SIGMA, true, false, CONTRA_DIR, 'STOP');
    y_go_cue_STOP(ni, :) = fr;
    
    [~, fr, ~] = calculate_psth_for_cell(cdata, 'go_cue', go_cue_epok, ...
        BIN_SIZE, SMOOTH_SIGMA, true, false, CONTRA_DIR, 'CONT');
    y_go_cue_CONT(ni, :) = fr;
    
    % Stop-signal aligned
    [~, fr, ~] = calculate_psth_for_cell(cdata, 'stop_cue', stop_sig_epok, ...
        BIN_SIZE, SMOOTH_SIGMA, true, false, CONTRA_DIR, 'GO');
    y_stop_sig_GO(ni, :) = fr;
    
    [~, fr, ~] = calculate_psth_for_cell(cdata, 'stop_cue', stop_sig_epok, ...
        BIN_SIZE, SMOOTH_SIGMA, true, false, CONTRA_DIR, 'STOP');
    y_stop_sig_STOP(ni, :) = fr;
    
    [~, fr, ~] = calculate_psth_for_cell(cdata, 'stop_cue', stop_sig_epok, ...
        BIN_SIZE, SMOOTH_SIGMA, true, false, CONTRA_DIR, 'CONT');
    y_stop_sig_CONT(ni, :) = fr;
end

%% Plot Figure 2 panels
alignment_labels = {'go_cue', 'stop_sig'};
alignment_titles = {'Go cue', 'Stop signal'};
x_axes = {sample_bins_gc, sample_bins_ss};
data_GO   = {y_go_cue_GO,   y_stop_sig_GO};
data_STOP = {y_go_cue_STOP, y_stop_sig_STOP};
data_CONT = {y_go_cue_CONT, y_stop_sig_CONT};

for ai = 1:2
    x = x_axes{ai};
    
    m_go   = nanmean(data_GO{ai}, 1);
    s_go   = nanstd(data_GO{ai}, 0, 1) / sqrt(size(data_GO{ai}, 1));
    m_stop = nanmean(data_STOP{ai}, 1);
    s_stop = nanstd(data_STOP{ai}, 0, 1) / sqrt(size(data_STOP{ai}, 1));
    m_cont = nanmean(data_CONT{ai}, 1);
    s_cont = nanstd(data_CONT{ai}, 0, 1) / sqrt(size(data_CONT{ai}, 1));
    
    figure('Position', [100 100 800 500]);
    hold on;
    
    % GO (green)
    fill([x, fliplr(x)], [m_go - s_go, fliplr(m_go + s_go)], ...
        [0.56 0.93 0.56], 'FaceAlpha', 0.3, 'EdgeColor', 'none');
    plot(x, m_go, 'Color', [0 0.5 0], 'LineWidth', 2, 'DisplayName', 'GO');
    
    % Correct STOP (black)
    fill([x, fliplr(x)], [m_stop - s_stop, fliplr(m_stop + s_stop)], ...
        [0.7 0.7 0.7], 'FaceAlpha', 0.3, 'EdgeColor', 'none');
    plot(x, m_stop, 'k', 'LineWidth', 2, 'DisplayName', 'Correct STOP');
    
    % CONT (blue)
    fill([x, fliplr(x)], [m_cont - s_cont, fliplr(m_cont + s_cont)], ...
        [0.68 0.85 0.90], 'FaceAlpha', 0.3, 'EdgeColor', 'none');
    plot(x, m_cont, 'b', 'LineWidth', 2, 'DisplayName', 'CONT');
    
    xline(0, 'r--', 'Alpha', 0.5, 'DisplayName', alignment_titles{ai});
    xlabel(sprintf('Time from %s (ms)', alignment_labels{ai}), 'FontSize', 12);
    ylabel('Firing rate (spikes/s)', 'FontSize', 12);
    title(sprintf('Population PSTH - %s aligned\n%d/%d MSN cells', ...
        alignment_titles{ai}, n_go, num_of_msn_cells), 'FontSize', 13);
    legend('Location', 'best');
    grid on; set(gca, 'GridAlpha', 0.3);
    hold off;
end

%% ========================================================================
%  4.1 Mean reaction times
%  ========================================================================

go_rt = cell_df(ismember(cell_df.cell_ID, go_neurons) & ...
    cell_df.trial_failed == false & cell_df.type == 'GO' & cell_df.dir == CONTRA_DIR, :);
mean_go_RT = nanmean(go_rt.reaction_time);

error_stop_rt = cell_df(ismember(cell_df.cell_ID, go_neurons) & ...
    cell_df.trial_failed == true & cell_df.type == 'STOP' & cell_df.dir == CONTRA_DIR, :);
mean_error_stop_RT = nanmean(error_stop_rt.reaction_time);

cont_rt = cell_df(ismember(cell_df.cell_ID, go_neurons) & ...
    cell_df.type == 'CONT' & cell_df.dir == CONTRA_DIR , :);
mean_cont_RT = nanmean(cont_rt.reaction_time);

fprintf('Mean GO RT: %.2f ms\n', mean_go_RT);
fprintf('Mean ERROR STOP RT: %.2f ms\n', mean_error_stop_RT);
fprintf('Mean CONT RT: %.2f ms\n', mean_cont_RT);


%% ========================================================================
%  Helper Functions
%  ========================================================================

function [baseline_rates, saccade_rates] = get_cell_spike_counts_condition_1( ...
        cell_data, contra_dir)
    % Get firing rates for baseline vs saccade windows on slow-GO contra trials
    %
    % Baseline: -300 to 0 ms from go_cue 
    % Saccade:  -100 to +100 ms from first_relevant_saccade
    
    % Filter: GO, contralateral, slow-GO, successful
    mask = (cell_data.type == 'GO') & ...
           cell_data.dir == contra_dir & ...
           cell_data.is_slow_go == 1 & ...
           cell_data.trial_failed == false;
    trials = cell_data(mask, :);
    
    n_trials = height(trials);
    baseline_rates = NaN(n_trials, 1);
    saccade_rates  = NaN(n_trials, 1);
    
    for ti = 1:n_trials
        spikes = trials.neural_data{ti};  % Raw spike times
        go_cue_time = trials.go_cue(ti);
        
        % Align to go_cue for baseline
        spikes_go_aligned = spikes - go_cue_time;
        baseline_count = sum(spikes_go_aligned >= -300 & spikes_go_aligned <= 0);
        baseline_rates(ti) = baseline_count / 0.3;  % Spikes/sec
        
        % Align to first_relevant_saccade for saccade window
        saccade_time = trials.first_relevant_saccade{ti};
        if iscell(saccade_time)
            saccade_time = saccade_time{1};
        end
        if isnumeric(saccade_time) && ~isempty(saccade_time) && ~isnan(saccade_time(1))
            s_time = saccade_time(1);  % First saccade
            spikes_sacc_aligned = spikes - s_time;
            saccade_count = sum(spikes_sacc_aligned >= -100 & spikes_sacc_aligned <= 100);
            saccade_rates(ti) = saccade_count / 0.2;  % Spikes/sec
        else
            saccade_rates(ti) = NaN;
        end
    end
    
    % Remove trials with NaN saccade
    valid = ~isnan(saccade_rates);
    baseline_rates = baseline_rates(valid);
    saccade_rates  = saccade_rates(valid);
end


function [slow_go_rates, correct_stop_rates] = get_cell_spike_counts_condition_2( ...
        cell_data, contra_dir)
    % Get firing rates for slow-GO vs correct-STOP in 0-400 ms from stop signal
    %
    % For GO trials: stop_cue is approximated as go_cue + mean_ssd
    
    % Compute mean SSD from this cell's STOP/CONT trials
    ssd_mask = (cell_data.type == 'STOP' | cell_data.type == 'CONT') & ...
               cell_data.dir == contra_dir & ...
               cell_data.trial_failed == false;
    mean_ssd = round(nanmean(cell_data.ssd_len(ssd_mask)));
    if isnan(mean_ssd)
        mean_ssd = 150;  % Default fallback
    end
    
    % --- Slow GO trials ---
    go_mask = (cell_data.type == 'GO') & ...
              cell_data.dir == contra_dir & ...
              cell_data.trial_failed == false & ...
              cell_data.is_slow_go == 1;
    go_trials = cell_data(go_mask, :);
    
    slow_go_rates = NaN(height(go_trials), 1);
    for ti = 1:height(go_trials)
        spikes = go_trials.neural_data{ti};
        % For GO trials, stop_cue ~ go_cue + mean_ssd
        stop_time = go_trials.go_cue(ti) + mean_ssd;
        spikes_aligned = spikes - stop_time;
        count = sum(spikes_aligned >= 0 & spikes_aligned <= 400);
        slow_go_rates(ti) = count / 0.4;  % Spikes/sec
    end
    
    % --- Correct STOP trials ---
    stop_mask = (cell_data.type == 'STOP') & ...
                cell_data.dir == contra_dir & ...
                cell_data.trial_failed == false;
    stop_trials = cell_data(stop_mask, :);
    
    correct_stop_rates = NaN(height(stop_trials), 1);
    for ti = 1:height(stop_trials)
        spikes = stop_trials.neural_data{ti};
        stop_time = stop_trials.stop_cue(ti);
        if isnan(stop_time)
            continue;
        end
        spikes_aligned = spikes - stop_time;
        count = sum(spikes_aligned >= 0 & spikes_aligned <= 400);
        correct_stop_rates(ti) = count / 0.4;  % Spikes/sec
    end
    
    % Remove NaN entries
    slow_go_rates = slow_go_rates(~isnan(slow_go_rates));
    correct_stop_rates = correct_stop_rates(~isnan(correct_stop_rates));
end


function [bin_centers, firing_rate, n_trials] = calculate_psth_for_cell( ...
        cell_data, alignment_point, epok, bin_size, smooth_sigma, ...
        success_only, failed_only, direction, trial_type)
    % Calculate PSTH for a single cell's trials
    %
    % Parameters:
    %   cell_data       - table of trials for one cell
    %   alignment_point - 'go_cue', 'stop_cue', 'first_relevant_saccade'
    %   epok            - [start, end] in ms
    %   bin_size        - bin width in ms
    %   smooth_sigma    - smoothing kernel duration (ms) for Pani SDF kernel
    %   success_only    - bool, only successful trials
    %   failed_only     - bool, only failed trials
    %   direction       - target direction (e.g. 0 or 180)
    %   trial_type      - 'GO', 'STOP', or 'CONT'
    
    % Filter trials
    mask = true(height(cell_data), 1);
    if ~isempty(trial_type)
        mask = mask & (cell_data.type == trial_type);
    end
    if ~isempty(direction)
        mask = mask & (cell_data.dir == direction);
    end
    if success_only
        mask = mask & (cell_data.trial_failed == false);
    end
    if failed_only
        mask = mask & (cell_data.trial_failed == true);
    end
    
    trials = cell_data(mask, :);
    n_trials = height(trials);
    
    if n_trials == 0
        bin_centers = [];
        firing_rate = [];
        return;
    end
    
    % Extended epoch to handle edge effects from smoothing
    ext_epok = [epok(1) - smooth_sigma, epok(2) + smooth_sigma];
    
    % Create bin edges
    bin_edges = ext_epok(1):bin_size:ext_epok(2);
    bin_centers_ext = bin_edges(1:end-1) + bin_size / 2;
    
    % Accumulate spikes
    spike_counts = zeros(1, numel(bin_centers_ext));
    
    for ti = 1:n_trials
        spikes = trials.neural_data{ti};
        
        % Get alignment time
        switch alignment_point
            case 'go_cue'
                align_time = trials.go_cue(ti);
            case 'stop_cue'
                if trials.type(ti) == 'GO'
                    % For GO trials, approximate stop_cue
                    ssd_vals = cell_data.ssd_len( ...
                        (cell_data.type == 'STOP' | cell_data.type == 'CONT') & ...
                        cell_data.trial_failed == false);
                    mean_ssd_val = round(nanmean(ssd_vals));
                    if isnan(mean_ssd_val); mean_ssd_val = 150; end
                    align_time = trials.go_cue(ti) + mean_ssd_val;
                else
                    align_time = trials.stop_cue(ti);
                end
            case 'first_relevant_saccade'
                sacc = trials.first_relevant_saccade{ti};
                if iscell(sacc); sacc = sacc{1}; end
                if ~isempty(sacc) && isnumeric(sacc) && ~isnan(sacc(1))
                    align_time = sacc(1);
                else
                    continue;  % Skip trial
                end
            otherwise
                align_time = 0;
        end
        
        if isnan(align_time)
            continue;
        end
        
        % Align spikes
        aligned_spikes = spikes - align_time;
        
        % Histogram
        counts = histcounts(aligned_spikes, bin_edges);
        spike_counts = spike_counts + counts;
    end
    
    % Convert to firing rate (spikes/sec)
    firing_rate_ext = (spike_counts / n_trials) / (bin_size / 1000);
    
    % Apply Pani SDF kernel (causal smoothing)
    kernel = pani_sdf_kernel(1.0, 20.0, smooth_sigma, 1.0);
    sdf_full = conv(firing_rate_ext, kernel, 'full');
    firing_rate_ext = sdf_full(1:numel(firing_rate_ext));  % Causal
    
    % Trim edges
    trim_mask = bin_centers_ext >= epok(1) & bin_centers_ext <= epok(2);
    bin_centers = bin_centers_ext(trim_mask);
    firing_rate = firing_rate_ext(trim_mask);
end


function kernel = pani_sdf_kernel(tau_g, tau_d, duration, dt)
    % Pani et al. 2022 spike density function kernel
    % K(t) = [1 - exp(-t/tau_g)] * exp(-t/tau_d)
    %
    % Parameters:
    %   tau_g    - growth time constant (ms), default 1.0
    %   tau_d    - decay time constant (ms), default 20.0
    %   duration - kernel duration (ms), default 20
    %   dt       - time step (ms), default 1.0
    %
    % Returns:
    %   kernel   - normalized kernel values
    
    t = 0:dt:(duration - dt);
    kernel = (1 - exp(-t / tau_g)) .* exp(-t / tau_d);
    kernel = kernel / sum(kernel);  % Normalize
end


function [reject, pvals_corrected] = holm_bonferroni(pvals, alpha)
    % Holm-Bonferroni multiple comparison correction
    %
    % Parameters:
    %   pvals  - vector of p-values
    %   alpha  - significance level (default 0.05)
    %
    % Returns:
    %   reject          - logical vector of rejected hypotheses
    %   pvals_corrected - adjusted p-values
    
    if nargin < 2; alpha = 0.05; end
    
    n = numel(pvals);
    [sorted_pvals, sort_idx] = sort(pvals);
    
    reject = false(n, 1);
    pvals_corrected = ones(n, 1);
    
    for i = 1:n
        adjusted_p = sorted_pvals(i) * (n - i + 1);
        pvals_corrected(sort_idx(i)) = min(adjusted_p, 1);
    end
    
    % Enforce monotonicity
    running_max = 0;
    for i = 1:n
        orig_i = sort_idx(i);
        pvals_corrected(orig_i) = max(pvals_corrected(orig_i), running_max);
        running_max = pvals_corrected(orig_i);
    end
    
    reject = pvals_corrected <= alpha;
end
