clear; clc;

fs = 128;
seg_len = 6 * fs;
subjects = 1:32;

accuracy_list = zeros(length(subjects), 1);
precision_list = zeros(length(subjects), 1);
recall_list = zeros(length(subjects), 1);
f1_list = zeros(length(subjects), 1);
CM_list = zeros(2,2,length(subjects));

for subj = subjects
    fprintf('\n--- Subject %d ---\n', subj);
    data_folder = "....Add DEAP dataset path here...";
    cd(data_folder);
    load(sprintf('s%02d.mat', subj));  % loads 'data', 'labels'

    baseline = mean(data(:, :, 1:fs*3), 3);
    data = data(:, :, fs*3+1:end);

    num_trials = size(data, 1);
    X = [];
    Y = [];

    for t = 1:num_trials
        eeg = double(squeeze(data(t, :, :)));
        eeg = bandpass(eeg', [4 45], fs)';
        n_segments = floor(size(eeg, 2) / seg_len);
        for seg = 1:n_segments
            seg_data = eeg(:, (seg-1)*seg_len + 1 : seg*seg_len);

            feat_sst_energy = zeros(32, 4);
            for ch = 1:32
                [s, f] = fsst(seg_data(ch,:), fs);
                s_mag = abs(s);
                feat_sst_energy(ch, 1) = mean(s_mag(f >= 4 & f < 8,:), 'all');
                feat_sst_energy(ch, 2) = mean(s_mag(f >= 8 & f < 13,:), 'all');
                feat_sst_energy(ch, 3) = mean(s_mag(f >= 13 & f < 30,:), 'all');
                feat_sst_energy(ch, 4) = mean(s_mag(f >= 30 & f < 45,:), 'all');
            end

            [pac_1,pac_2,pac_3,pac_4,pac_5,pac_6] = deal(zeros(32,1)); 
            [aac_1,aac_2,aac_3,aac_4,aac_5,aac_6] = deal(zeros(32,1));
            for ch = 1:32
                theta = bandpass(seg_data(ch,:), [4 8], fs);
                alpha = bandpass(seg_data(ch,:), [8 13], fs);
                beta = bandpass(seg_data(ch,:), [13 30], fs);
                gamma = bandpass(seg_data(ch,:), [30 45], fs);

                theta_phase = angle(hilbert(theta));
                alpha_phase = angle(hilbert(alpha));
                beta_phase = angle(hilbert(beta));
                gamma_phase = angle(hilbert(gamma));
                theta_amp = abs(hilbert(theta));
                alpha_amp = abs(hilbert(alpha));
                beta_amp = abs(hilbert(beta));
                gamma_amp = abs(hilbert(gamma));

                pac_1(ch) = abs(mean(theta_phase .* exp(1i * alpha_amp)));
                pac_2(ch) = abs(mean(theta_phase .* exp(1i * beta_amp)));
                pac_3(ch) = abs(mean(theta_phase .* exp(1i * gamma_amp)));
                pac_4(ch) = abs(mean(alpha_phase .* exp(1i * beta_amp)));
                pac_5(ch) = abs(mean(alpha_phase .* exp(1i * gamma_amp)));               
                pac_6(ch) = abs(mean(beta_phase .* exp(1i * gamma_amp)));

                aac_1(ch) = abs(corr(theta', alpha', 'Type','Pearson'));
                aac_2(ch) = abs(corr(theta', beta', 'Type','Pearson'));
                aac_3(ch) = abs(corr(theta', gamma', 'Type','Pearson'));
                aac_4(ch) = abs(corr(alpha', beta', 'Type','Pearson'));
                aac_5(ch) = abs(corr(alpha', gamma', 'Type','Pearson'));
                aac_6(ch) = abs(corr(beta', gamma', 'Type','Pearson'));
            end

            feat_pac = [pac_1,pac_2,pac_3,pac_4,pac_5,pac_6];
            feat_aac = [aac_1,aac_2,aac_3,aac_4,aac_5,aac_6];

            combined_features = [feat_sst_energy, feat_pac, feat_aac];
            %X = cat(1, X, combined_features);
            if size(combined_features, 2) == 16
                X = cat(1, X, combined_features);
                Y = [Y; labels(t, 1) >= 5];       % Valence: 1 & Arousal: 2
            else
                warning('Skipping segment due to feature size mismatch.');
            end
            %Y = [Y; labels(t, 1) >= 5];
        end
    end

    % Reshape for CNN input [trial, 32, 38]
    num_samples = size(X, 1);
    X = reshape(X, [], 32, 16);
    X = permute(X, [2,3,4,1]);  % [32,38,1,N]
   Y = categorical(Y); % Binary classification labels
    
   K = 5;
   cv = cvpartition(Y, 'KFold', K);

    acc_fold = zeros(K,1);
    prec_fold = zeros(K,1);
    rec_fold = zeros(K,1);
    f1_fold = zeros(K,1);
    cm_fold = zeros(2,2,5);

    for k = 1:K
        train_idx = training(cv, k);
        test_idx = test(cv, k);

        XTrain = X(:,:,:,train_idx);
        YTrain = Y(train_idx);
        XTest = X(:,:,:,test_idx);
        YTest = Y(test_idx);

        % --- Define CNN + LSTM layers ---
        layers = [
            imageInputLayer([32 16 1], 'Name', 'input')
            convolution2dLayer([3 2], 16, 'Padding', 'same')
            batchNormalizationLayer
            reluLayer
            maxPooling2dLayer([2 2])
            convolution2dLayer([3 2], 32, 'Padding', 'same')
            batchNormalizationLayer
            reluLayer
            flattenLayer('Name','flatten')
            lstmLayer(128, 'OutputMode','last','Name','lstm')
            fullyConnectedLayer(64, 'Name','fc1')
            reluLayer('Name','relu3')
            fullyConnectedLayer(2, 'Name','fc2')
            softmaxLayer('Name','softmax')
            classificationLayer('Name','output')
            ];
        
    

        % --- Training options ---
        options = trainingOptions('adam', ...
            'MaxEpochs', 10, ...
            'MiniBatchSize', 16, ...
            'Shuffle', 'every-epoch', ...
            'Verbose', false);
    
        % --- Train network ---
        net = trainNetwork(XTrain, YTrain, layers, options);
    
        % --- Test network ---
        YPred = classify(net, XTest);

        acc = mean(YPred == YTest);
        cm = confusionmat(YTest, YPred);
        tp = cm(2,2); fp = cm(1,2); fn = cm(2,1);

        prec = tp / (tp + fp + eps);
        rec = tp / (tp + fn + eps);
        f1 = 2 * (prec * rec) / (prec + rec + eps);

        acc_fold(k) = acc;
        prec_fold(k) = prec;
        rec_fold(k) = rec;
        f1_fold(k) = f1;
        cm_fold(:,:,k) = cm;
    end

    accuracy_list(subj) = mean(acc_fold);
    precision_list(subj) = mean(prec_fold);
    recall_list(subj) = mean(rec_fold);
    f1_list(subj) = mean(f1_fold);
    CM_list(:,:,subj) = mean(cm_fold,3);

    fprintf('Mean Accuracy : %.2f%%\n', accuracy_list(subj)*100);
    fprintf('Mean Precision: %.2f%%\n', precision_list(subj)*100);
    fprintf('Mean Recall   : %.2f%%\n', recall_list(subj)*100);
    fprintf('Mean F1 Score : %.2f%%\n', f1_list(subj)*100);
end

%% Final Report
fprintf('\n=== CNN+LSTM with 5-Fold CV (SST + PAC + AAC) ===\n');
fprintf('Avg Accuracy : %.2f%%\n', mean(accuracy_list)*100);
fprintf('Avg Precision: %.2f%%\n', mean(precision_list)*100);
fprintf('Avg Recall   : %.2f%%\n', mean(recall_list)*100);
fprintf('Avg F1 Score : %.2f%%\n', mean(f1_list)*100);
disp('Avg Confusion Matrix:');
disp(mean(CM_list,3));

