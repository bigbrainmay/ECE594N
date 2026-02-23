clear
load('for_Gabe_v3.mat')

% fulltrace or Snips
trace_type = 'fulltrace';
% Angle or grey (if 'Snips')
mode = 'grey';
signal_length = 1000; % length to downsample/interpolate to
win = 20; % averaging window size
% mode = 'grey';

% absolutely hideous way of extracting spine structs
if strcmp(trace_type,'Snips')
    spines = cellfun( @(q) [q.spine] ,arrayfun(@(p) p.Snips, ...
        C, 'UniformOutput', false), 'UniformOutput', false);
    spines = cellfun(@(m) [m.(mode)]',spines,'UniformOutput',false);
else
    spines = arrayfun(@(p) p.fulltrace, ...
        C, 'UniformOutput', false);
end

% Number of elements
n = numel(spines);

% Preallocate cell array
proc = cell(n,1);
nzs = [];

% Filter design
b = ones(1,win)/win;   % moving average

for i = 1:n
    A_raw = spines{i};  
    rlg = C(i).ret_los_gain;

    nonzero_rows = ~all(A_raw == 0,2);
    nzs = [nzs; nonzero_rows];

    A_raw = A_raw(nonzero_rows,:);

    % zero-phase low-pass filter
    A_filt = filtfilt(b,1,A_raw')';

    % downsample/interp
    orig_n = size(A_filt,2);
    x = linspace(1,signal_length,orig_n);
    A_short = interp1(x,A_filt.',1:signal_length).';

    proc{i} = A_short;

%     plot(A_raw(1,:)); hold on
%     plot(A_filt(1,:));
%     plot(A_short(1,:));
%     legend raw filtered downsampled

end

rlg_all = type(find(nzs));
tuned_all = tuned(find(nzs));
VR_all = VR(find(nzs));

% Vertically concatenate
traces = vertcat(proc{:});

save('processed_traces.mat',"traces", "rlg_all", "tuned_all","VR_all")