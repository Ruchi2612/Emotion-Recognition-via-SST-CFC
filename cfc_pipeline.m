clear; clc;

fs = 128;
segment_len = 6 * fs;  % 6 seconds
subjects = 1:32;

accuracy_list = zeros(length(subjects), 1);
precision_list = zeros(length(subjects), 1);
recall_list = zeros(length(subjects), 1);
f1_list = zeros(length(subjects), 1);
CM_list = zeros(2,2,length(subjects));

for subj = 1:32
    fprintf('\n--- Subject %d ---\n', subj);
    data_folder = ("....Add dataset path here...");
    cd(data_folder);
    load(sprintf('s%02d.mat', subj));  % loads 'data', 'labels'

    num_trials = size(data, 1);
    features_all = [];
    labels_all = [];

    for t = 1:num_trials
        eeg = double(squeeze(data(t, :, :)));  % [32, 8064]

        % Baseline removal using first 3 sec (384 samples)
        baseline = mean(eeg(:, 1:384), 2);
        eeg = eeg(:, 385:end) - baseline;

        eeg = bandpass(eeg', [4 45], fs)';  % bandpass filtering

        total_len = size(eeg, 2);
        num_segments = floor(total_len / segment_len);

        for seg = 1:num_segments
            seg_start = (seg - 1) * segment_len + 1;
            seg_end = seg_start + segment_len - 1;
            segment = eeg(:, seg_start:seg_end);

            % ---CFC Feature Extraction ---
            [pac_1,pac_2,pac_3,pac_4,pac_5,pac_6] = deal(zeros(32,1)); 
            [aac_1,aac_2,aac_3,aac_4,aac_5,aac_6] = deal(zeros(32,1));
            
            for ch = 1:32
                theta = bandpass(segment(ch,:), [4 8], fs);
                alpha = bandpass(segment(ch,:), [8 13], fs);
                beta = bandpass(segment(ch,:), [13 30], fs);
                gamma = bandpass(segment(ch,:), [30 45], fs);

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

            % Combine CFC (PAC + AAC)
            feat_pac = [pac_1,pac_2,pac_3,pac_4,pac_5,pac_6,pac_7,pac_8,pac_9,pac_10,pac_11,pac_12];
            feat_aac = [aac_1,aac_2,aac_3,aac_4,aac_5,aac_6];
            combined_features = [feat_pac(:); feat_aac(:)];  % [32x12 → 384x1]
            features_all = [features_all; combined_features'];
            labels_all = [labels_all; labels(t, 1) >= 5];  % Valence: 1 & Arousal: 2
        end
    end
    %% 5-Fold Cross Validation (Subject-Dependent)
    labels_all = categorical(labels_all);
    cv = cvpartition(labels_all, 'KFold', 5);

    acc_fold = zeros(5,1);
    prec_fold = zeros(5,1);
    rec_fold = zeros(5,1);
    f1_fold = zeros(5,1);
    CM_fold = zeros(2,2,5);

    for fold = 1:5
        train_idx = training(cv, fold);
        test_idx = test(cv, fold);

        XTrain = features_all(train_idx, :);
        YTrain = labels_all(train_idx);
        XTest = features_all(test_idx, :);
        YTest = labels_all(test_idx);

        % Train SVM/KNN/RF (Choose classifier)
        SVMModel = fitcsvm(XTrain, YTrain, 'KernelFunction', 'rbf', 'Standardize', true);
        %KNNModel = fitcknn(XTrain, YTrain, 'NumNeighbors',3, 'NSMethod','kdtree','Distance','euclidean','Standardize', 1);
        %RFModel = fitcensemble(XTrain, YTrain, 'Method','Bag','Learners', 'tree', 'NumLearningCycles',10);%'MaxNumSplits', n-1);

        % Predict and Evaluate
        YPred = predict(SVMModel, XTest);
        acc_fold(fold) = mean(YPred == YTest);

        confMat = confusionmat(YTest, YPred);
        tp = confMat(2,2); fp = confMat(1,2); fn = confMat(2,1);

        prec_fold(fold) = tp / (tp + fp + eps);
        rec_fold(fold)  = tp / (tp + fn + eps);
        f1_fold(fold)   = 2 * (prec_fold(fold) * rec_fold(fold)) / (prec_fold(fold) + rec_fold(fold) + eps);
        CM_fold(:,:,fold) = confMat;
    end

    % Average across 5 folds
    accuracy_list(subj) = mean(acc_fold);
    precision_list(subj) = mean(prec_fold);
    recall_list(subj) = mean(rec_fold);
    f1_list(subj) = mean(f1_fold);
    CM_list(:,:,subj) = mean(CM_fold,3);

    fprintf('Mean Accuracy  : %.2f%%\n', accuracy_list(subj) * 100);
    fprintf('Mean Precision : %.2f%%\n', precision_list(subj) * 100);
    fprintf('Mean Recall    : %.2f%%\n', recall_list(subj) * 100);
    fprintf('Mean F1 Score  : %.2f%%\n', f1_list(subj) * 100);
    %disp('Mean Confusion Matrix:'); disp(CM_list(:,:,subj));
end

%% Summary
fprintf('\n=== Subject-Dependent SVM (5-fold CV) Summary ===\n');
fprintf('Avg Accuracy  : %.2f%%\n', mean(accuracy_list) * 100);
fprintf('Avg Precision : %.2f%%\n', mean(precision_list) * 100);
fprintf('Avg Recall    : %.2f%%\n', mean(recall_list) * 100);
fprintf('Avg F1 Score  : %.2f%%\n', mean(f1_list) * 100);
disp('Avg Confusion Matrix:');
disp(mean(CM_list,3));