function [InternalTable, ExternalTable, TimeTable, OptimizedParamsTable, StatTestTable, OptimizedParamsRaw, ML_Probs_Int, ML_Labels_Int, ML_Probs_Ext, ML_Labels_Ext, extRes, timeRes, BestModels, PlotData, PlotFunc] = ML_Repeated_Nested_CV(data, K, holdPercent, nIter, selectedModels, appLogFunc)
% ML_REPEATED_NESTED_CV - Leakage-Free Repeated Nested Cross-Validation
% Features: Fold-local imputation, global Z-score scaling, hyperparameter tuning

if isempty(gcp('nocreate'))
    parpool; 
end
rng(42);
holdRatio = holdPercent / 100;

% READ DATA
if istable(data)
    Xraw = table2array(data(:, 1:end-1)); 
    y = table2array(data(:, end));
else
    Xraw = data(:, 1:end-1); 
    y = data(:, end); 
end
y = double(y);               

validIdx = ~isnan(y);
Xraw = Xraw(validIdx, :);
y = y(validIdx);

uClass = unique(y); 
numC = numel(uClass);
isMultiClass = numC > 2;

if nargin < 5 || isempty(selectedModels)
    models = {'DT', 'LR', 'DA', 'KNN', 'SVM', 'NB', 'NN', 'ENS', 'GAM', 'GLM'};
else
    models = selectedModels; 
end
nModels = numel(models);

intRes = zeros(nModels, nIter, 8); 
extRes = zeros(nModels, nIter, 8); 
timeRes = zeros(nModels, nIter, 2); 
OptimizedParamsRaw = cell(nModels, nIter);
BestModels = struct(); 
bestModelScore = -inf(1, nModels);

ML_Probs_Int = cell(1, nModels); 
ML_Labels_Int = cell(1, nModels); 
ML_Preds_Int = cell(1, nModels);
ML_Probs_Ext = cell(1, nModels); 
ML_Labels_Ext = cell(1, nModels); 
ML_Preds_Ext = cell(1, nModels);

totalETA_Steps = nIter * nModels * (K + 1);
currentETA_Step = 0;
globalTimer = tic;

for iter = 1:nIter
    rng(iter, 'twister');
    cvHold = cvpartition(y, 'HoldOut', holdRatio, 'Stratify', true);
    Xtr_full_raw = Xraw(training(cvHold), :); 
    ytr_full = y(training(cvHold));
    Xte_raw = Xraw(test(cvHold), :); 
    yte = y(test(cvHold));

    cv = cvpartition(ytr_full, 'KFold', K);
    
    for m = 1:nModels
        modelName = models{m};
        
        elapsed = toc(globalTimer);
        if currentETA_Step == 0
            remStr = "Calculating...";
        else
            avgTime = elapsed / currentETA_Step;
            remSec = avgTime * (totalETA_Steps - currentETA_Step);
            remStr = sprintf('~%dm %ds', floor(remSec/60), floor(mod(remSec, 60)));
        end
        
        if nargin >= 6 && ~isempty(appLogFunc)
            appLogFunc(sprintf('Iteration %d/%d | Model: %s | Outer Opt. | ETA: %s', iter, nIter, modelName, remStr)); 
        end
        currentETA_Step = currentETA_Step + 1;
        
        t_train = tic; 
        prep_full = fitPreprocessRules(Xtr_full_raw);
        Xtr_full = applyPreprocessRules(Xtr_full_raw, prep_full);
        Xte = applyPreprocessRules(Xte_raw, prep_full);
        
        innerHoldout = cvpartition(ytr_full, 'HoldOut', 0.20, 'Stratify', true);
        opt = struct('MaxObjectiveEvaluations', 30, 'CVPartition', innerHoldout, 'ShowPlots', false, 'Verbose', 0, 'UseParallel', true);
        
        mdl_full = trainModel(modelName, Xtr_full, ytr_full, opt, isMultiClass); 
        timeRes(m, iter, 1) = toc(t_train);
        
        try 
            OptimizedParamsRaw{m, iter} = mdl_full.HyperparameterOptimizationResults.XAtMinObjective; 
        catch 
            OptimizedParamsRaw{m, iter} = table(); 
        end
        
        t_test = tic; 
        if isa(mdl_full, 'classreg.learning.classif.ClassificationECOC') || isa(mdl_full, 'classreg.learning.classif.CompactClassificationECOC')
            try
                [ypred_ext_lbl, ~, ~, score_ext] = predict(mdl_full, Xte);
                if isempty(score_ext), [ypred_ext_lbl, score_ext] = predict(mdl_full, Xte); end
            catch
                [ypred_ext_lbl, score_ext] = predict(mdl_full, Xte);
            end
        elseif isa(mdl_full, 'GeneralizedLinearModel') 
            prob1 = predict(mdl_full, Xte);
            score_ext = [1-prob1, prob1];
            ypred_ext_lbl = min(uClass) * ones(size(prob1));
            ypred_ext_lbl(prob1 >= 0.5) = max(uClass);
        else
            [ypred_ext_lbl, score_ext] = predict(mdl_full, Xte);
        end
        timeRes(m, iter, 2) = toc(t_test);
        
        if isMultiClass
            prob_ext = alignScoreMatrix(score_ext, mdl_full, uClass); 
        else
            prob_ext = positiveClassScore(score_ext, mdl_full, uClass); 
        end
        
        pred_internal_labels = zeros(length(ytr_full), 1);
        if isMultiClass
            prob_internal = zeros(length(ytr_full), numC);
        else
            prob_internal = zeros(length(ytr_full), 1);
        end

        for k = 1:K
            elapsed = toc(globalTimer);
            avgTime = elapsed / currentETA_Step;
            remSec = avgTime * (totalETA_Steps - currentETA_Step);
            remStr = sprintf('~%dm %ds', floor(remSec/60), floor(mod(remSec, 60)));
            
            if nargin >= 6 && ~isempty(appLogFunc)
                appLogFunc(sprintf('Iteration %d/%d | Model: %s | Inner Fold %d/%d | ETA: %s', iter, nIter, modelName, k, K, remStr)); 
            end
            currentETA_Step = currentETA_Step + 1;
            
            valIdx = test(cv, k);
            trIdx = training(cv, k);
            
            Xtr_k_raw = Xtr_full_raw(trIdx, :);
            ytr_k = ytr_full(trIdx);
            Xval_k_raw = Xtr_full_raw(valIdx, :);
            
            prep_k = fitPreprocessRules(Xtr_k_raw);
            Xtr_k = applyPreprocessRules(Xtr_k_raw, prep_k);
            Xval_k = applyPreprocessRules(Xval_k_raw, prep_k);
            
            innerHoldout_k = cvpartition(ytr_k, 'HoldOut', 0.20, 'Stratify', true);
            opt_k = struct('MaxObjectiveEvaluations', 30, 'CVPartition', innerHoldout_k, 'ShowPlots', false, 'Verbose', 0, 'UseParallel', true);
            
            foldMdl = trainModel(modelName, Xtr_k, ytr_k, opt_k, isMultiClass);
            
            if isa(foldMdl, 'classreg.learning.classif.CompactClassificationSVM') || isa(foldMdl, 'classreg.learning.classif.ClassificationSVM')
                ws = warning('query','all'); warning('off','all');
                try foldMdl = fitPosterior(foldMdl, Xtr_k, ytr_k); catch; end
                warning(ws);
            end
            
            if isa(foldMdl, 'classreg.learning.classif.CompactClassificationECOC') || isa(foldMdl, 'classreg.learning.classif.ClassificationECOC')
                try
                    [lbl, ~, ~, post] = predict(foldMdl, Xval_k);
                    if isempty(post), [lbl, post] = predict(foldMdl, Xval_k); end
                catch
                    [lbl, post] = predict(foldMdl, Xval_k);
                end
                scr = post;
            elseif isa(foldMdl, 'GeneralizedLinearModel')
                prob1 = predict(foldMdl, Xval_k);
                scr = [1-prob1, prob1];
                lbl = min(uClass) * ones(size(prob1));
                lbl(prob1 >= 0.5) = max(uClass);
            else
                [lbl, scr] = predict(foldMdl, Xval_k);
            end
            
            pred_internal_labels(valIdx) = lbl;
            if isMultiClass
                prob_internal(valIdx, :) = alignScoreMatrix(scr, foldMdl, uClass); 
            else
                prob_internal(valIdx) = positiveClassScore(scr, foldMdl, uClass); 
            end
        end

        intRes(m, iter, :) = calculateMetrics(ytr_full, pred_internal_labels, prob_internal, isMultiClass, uClass, []);
        
        if ~isMultiClass
            thr_int = intRes(m, iter, 8);
            pred_internal_labels = min(uClass) * ones(size(prob_internal));
            pred_internal_labels(prob_internal >= thr_int) = max(uClass);
            
            ypred_ext_lbl = min(uClass) * ones(size(prob_ext));
            ypred_ext_lbl(prob_ext >= thr_int) = max(uClass);
        else
            thr_int = NaN;
        end

        extRes(m, iter, :) = calculateMetrics(yte, ypred_ext_lbl, prob_ext, isMultiClass, uClass, thr_int);
        
        ML_Probs_Int{m} = [ML_Probs_Int{m}; prob_internal]; 
        ML_Labels_Int{m} = [ML_Labels_Int{m}; ytr_full]; 
        ML_Preds_Int{m} = [ML_Preds_Int{m}; pred_internal_labels];
        
        ML_Probs_Ext{m} = [ML_Probs_Ext{m}; prob_ext]; 
        ML_Labels_Ext{m} = [ML_Labels_Ext{m}; yte]; 
        ML_Preds_Ext{m} = [ML_Preds_Ext{m}; ypred_ext_lbl];
        
        currentScore = intRes(m, iter, 2); 
        if isnan(currentScore)
            currentScore = intRes(m, iter, 1); 
        end
        
        if currentScore > bestModelScore(m) || iter == 1
            bestModelScore(m) = currentScore; 
            BestModels.(modelName) = mdl_full; 
            BestModels.([modelName '_Preprocess']) = prep_full;
            BestModels.([modelName '_TestX']) = Xte; 
            BestModels.([modelName '_TrainX']) = Xtr_full;
        end
    end
end

meanInt = mean(intRes, 2, 'omitnan'); stdInt = std(intRes, 0, 2, 'omitnan');
meanExt = mean(extRes, 2, 'omitnan'); stdExt = std(extRes, 0, 2, 'omitnan');
meanTime = mean(timeRes, 2, 'omitnan'); stdTime = std(timeRes, 0, 2, 'omitnan');

varNames = {'Model', 'Acc_mean', 'Acc_std', 'AUC_mean', 'AUC_std', 'Sens_mean', 'Sens_std', 'Spec_mean', 'Spec_std', 'F1_mean', 'F1_std', 'MCC_mean', 'MCC_std', 'Brier_mean', 'Brier_std', 'Thr_mean', 'Thr_std'};
InternalTable = table(models', meanInt(:,1), stdInt(:,1), meanInt(:,2), stdInt(:,2), meanInt(:,3), stdInt(:,3), meanInt(:,4), stdInt(:,4), meanInt(:,5), stdInt(:,5), meanInt(:,6), stdInt(:,6), meanInt(:,7), stdInt(:,7), meanInt(:,8), stdInt(:,8), 'VariableNames', varNames);
ExternalTable = table(models', meanExt(:,1), stdExt(:,1), meanExt(:,2), stdExt(:,2), meanExt(:,3), stdExt(:,3), meanExt(:,4), stdExt(:,4), meanExt(:,5), stdExt(:,5), meanExt(:,6), stdExt(:,6), meanExt(:,7), stdExt(:,7), meanExt(:,8), stdExt(:,8), 'VariableNames', varNames);
TimeTable = table(models', meanTime(:,:,1), stdTime(:,:,1), meanTime(:,:,2), stdTime(:,:,2), 'VariableNames', {'Model', 'TrainTime_mean', 'TrainTime_std', 'TestTime_mean', 'TestTime_std'});

optParamsCell = cell(nModels, 1);
for m = 1:nModels
    modelName = models{m};
    try
        mdl = BestModels.(modelName);
        if isprop(mdl, 'Learners') && ~isempty(mdl.Learners)
            mp = mdl.Learners{1}.ModelParameters;
        elseif isprop(mdl, 'ModelParameters')
            mp = mdl.ModelParameters;
        else
            mp = [];
        end
        
        strParam = "";
        if ~isempty(mp)
            props = properties(mp);
            for i = 1:length(props)
                val = mp.(props{i});
                if isnumeric(val) && isscalar(val)
                    strParam = strParam + string(props{i}) + ": " + num2str(val, '%.4f') + " | ";
                elseif ischar(val) || (isstring(val) && isscalar(val))
                    strParam = strParam + string(props{i}) + ": " + string(val) + " | ";
                end
            end
        end
        if strlength(strParam) > 0
            optParamsCell{m} = char(strParam);
        else
            optParamsCell{m} = 'Default Parameters';
        end
    catch
        optParamsCell{m} = 'N/A'; 
    end
end
OptimizedParamsTable = table(models', optParamsCell, 'VariableNames', {'Model', 'Best_Hyperparameters'});

pValsStr = strings(nModels, nModels); 
for i = 1:nModels
    for j = 1:nModels
        if i == j
            pValsStr(i,j) = "-"; 
        else
            try 
                p = signrank(squeeze(extRes(i,:,1)), squeeze(extRes(j,:,1))); 
                if p < 0.05
                    pValsStr(i,j) = sprintf('%.4f (*)', p); 
                else
                    pValsStr(i,j) = sprintf('%.4f', p); 
                end
            catch
                pValsStr(i,j) = "NaN"; 
            end
        end
    end
end
StatTestTable = array2table(pValsStr, 'VariableNames', models, 'RowNames', models);

PlotData.models = models; PlotData.numC = numC; PlotData.uClass = uClass;
PlotData.ML_Labels_Int = ML_Labels_Int; PlotData.ML_Probs_Int = ML_Probs_Int; PlotData.ML_Preds_Int = ML_Preds_Int;
PlotData.ML_Labels_Ext = ML_Labels_Ext; PlotData.ML_Probs_Ext = ML_Probs_Ext; PlotData.ML_Preds_Ext = ML_Preds_Ext;
PlotData.meanInt = meanInt; PlotData.stdInt = stdInt; PlotData.meanExt = meanExt; PlotData.stdExt = stdExt;
PlotData.intRes = intRes; PlotData.extRes = extRes;
PlotFunc = @visualizeOutputs;

end

function prep = fitPreprocessRules(X)
    medVals = median(X, 1, 'omitnan');
    medVals(isnan(medVals)) = 0; 
    
    X_imp = X;
    for i = 1:size(X, 2)
        idx = isnan(X_imp(:, i));
        X_imp(idx, i) = medVals(i);
    end

    stdX = std(X_imp, 0, 1);
    keepStd = ~(stdX == 0 | isnan(stdX));
    keepIdx = find(keepStd);
    if isempty(keepIdx), error('No features left after preprocessing.'); end
    Xtmp = X_imp(:, keepIdx);
    
    if size(Xtmp, 2) > 1
        R = corr(Xtmp, 'Rows', 'pairwise'); 
        R(isnan(R)) = 0; 
        [~, colIdx] = find(abs(tril(R, -1)) > 0.9999);
        if ~isempty(colIdx)
            keepTmp = true(1, size(Xtmp, 2));
            keepTmp(unique(colIdx)) = false;
            keepIdx = keepIdx(keepTmp);
            Xtmp = X_imp(:, keepIdx); 
        end
    end
    
    mu = mean(Xtmp, 1);
    sigma = std(Xtmp, 0, 1);
    sigma(sigma == 0) = 1; 
    
    prep = struct('medVals', medVals, 'keepIdx', keepIdx, 'mu', mu, 'sigma', sigma, 'nOriginalFeatures', size(X, 2));
end

function Xp = applyPreprocessRules(X, prep)
    Xp_imp = X;
    for i = 1:size(X, 2)
        idx = isnan(Xp_imp(:, i));
        Xp_imp(idx, i) = prep.medVals(i);
    end
    
    Xp = Xp_imp(:, prep.keepIdx);
    Xp = (Xp - prep.mu) ./ prep.sigma;
end

function scoreAligned = alignScoreMatrix(score, mdl, uClass)
    numSamples = size(score, 1);
    numC = numel(uClass);
    scoreAligned = zeros(numSamples, numC); 
    
    try
        cls = mdl.ClassNames;
        [tf, loc] = ismember(uClass, cls);
        for i = 1:numC
            if tf(i) && loc(i) > 0 && loc(i) <= size(score, 2)
                scoreAligned(:, i) = score(:, loc(i));
            end
        end
    catch
        cLen = min(size(score, 2), numC);
        scoreAligned(:, 1:cLen) = score(:, 1:cLen);
    end
    
    if max(scoreAligned(:)) > 1.01 || min(scoreAligned(:)) < -0.01
        e = exp(scoreAligned - max(scoreAligned, [], 2)); 
        scoreAligned = e ./ sum(e, 2);
    end
end

function p = positiveClassScore(score, mdl, uClass)
    pCls = max(uClass);
    p = score(:, end);
    try
        cls = mdl.ClassNames;
        [tf, loc] = ismember(pCls, cls);
        if tf && loc <= size(score, 2)
            p = score(:, loc);
        end
    catch
    end
    
    if max(p) > 1.01 || min(p) < -0.01
        p = 1 ./ (1 + exp(-p));
    end
end

function mdl = trainModel(modelName, X, y, opt, isMultiClass)
ws = warning('query','all'); warning('off','all');
    switch modelName
        case 'DT'
            mdl = fitctree(X, y, 'OptimizeHyperparameters', 'auto', 'HyperparameterOptimizationOptions', opt);
        case 'LR'
            if isMultiClass
                t = templateLinear('Learner', 'logistic');
                mdl = fitcecoc(X, y, 'Learners', t, 'FitPosterior', 1, 'OptimizeHyperparameters', 'auto', 'HyperparameterOptimizationOptions', opt);
            else
                mdl = fitclinear(X, y, 'Learner', 'logistic', 'OptimizeHyperparameters', 'auto', 'HyperparameterOptimizationOptions', opt);
            end
        case 'DA'
            mdl = fitcdiscr(X, y, 'OptimizeHyperparameters', 'auto', 'HyperparameterOptimizationOptions', opt);
        case 'KNN'
            mdl = fitcknn(X, y, 'OptimizeHyperparameters', 'auto', 'HyperparameterOptimizationOptions', opt);
        case 'SVM'
            if isMultiClass
                t = templateSVM(); 
                mdl = fitcecoc(X, y, 'Learners', t, 'FitPosterior', 1, 'OptimizeHyperparameters', 'auto', 'HyperparameterOptimizationOptions', opt);
            else
                mdl = fitcsvm(X, y, 'OptimizeHyperparameters', 'auto', 'HyperparameterOptimizationOptions', opt); 
                mdl = fitPosterior(mdl, X, y);
            end
        case 'NB'
            hasZeroVar = false;
            uC = unique(y);
            for ci = 1:numel(uC)
                cls_data = X(y == uC(ci), :);
                if ~isempty(cls_data) && any(var(cls_data, 0, 1) == 0)
                    hasZeroVar = true;
                    break;
                end
            end
            
            if hasZeroVar
                mdl = fitcnb(X, y, 'DistributionNames', 'kernel', 'OptimizeHyperparameters', {'Width'}, 'HyperparameterOptimizationOptions', opt);
            else
                mdl = fitcnb(X, y, 'OptimizeHyperparameters', 'auto', 'HyperparameterOptimizationOptions', opt);
            end
        case 'NN'
            mdl = fitcnet(X, y, 'OptimizeHyperparameters', 'auto', 'HyperparameterOptimizationOptions', opt);
        case 'ENS'
            mdl = fitcensemble(X, y, 'OptimizeHyperparameters', 'auto', 'HyperparameterOptimizationOptions', opt);
        case 'GAM'
            mdl = fitcgam(X, y, 'OptimizeHyperparameters', 'auto', 'HyperparameterOptimizationOptions', opt);
        case 'GLM'
            if isMultiClass
                t = templateLinear('Learner', 'logistic');
                mdl = fitcecoc(X, y, 'Learners', t, 'FitPosterior', 1);
            else
                mdl = fitglm(X, y, 'Distribution', 'binomial', 'Link', 'logit');
            end
    end
    warning(ws);
end

function res = calculateMetrics(ytrue, ypred_lbl, prob, isMultiClass, uClass, fixedThr)
    numC = numel(uClass); 
    pCls = max(uClass); 
    nCls = min(uClass);
    if ~isMultiClass
        try 
            if isempty(fixedThr)
                [Xr, Yr, T, AUC] = perfcurve(ytrue, prob, pCls); 
                [~, idx] = max(Yr - Xr); 
                thr = T(idx); 
            else
                thr = fixedThr;
                [~, ~, ~, AUC] = perfcurve(ytrue, prob, pCls);
            end
            yp = prob >= thr;
            
            TP = sum(yp == 1 & ytrue == pCls); 
            TN = sum(yp == 0 & ytrue == nCls); 
            FP = sum(yp == 1 & ytrue == nCls); 
            FN = sum(yp == 0 & ytrue == pCls);
            Acc = (TP+TN)/(TP+TN+FP+FN); 
            Sens = TP/(TP+FN+eps); 
            Spec = TN/(TN+FP+eps); 
            F1 = 2*TP/(2*TP+FP+FN+eps); 
            MCC = (TP*TN-FP*FN)/sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)+eps); 
            Brier = mean((prob-(ytrue==pCls)).^2,'omitnan');
        catch
            [Acc,AUC,Sens,Spec,F1,MCC,Brier,thr] = deal(NaN); 
        end
    else
        try 
            Acc = sum(ypred_lbl==ytrue)/length(ytrue); 
            [SA, SpA, F1A, AUCA, MCCA] = deal(zeros(1, numC)); 
            yoh = zeros(length(ytrue), numC);
            for c = 1:numC
                yoh(:,c) = (ytrue == uClass(c)); 
                TPc = sum(ypred_lbl==uClass(c) & ytrue==uClass(c)); 
                TNc = sum(ypred_lbl~=uClass(c) & ytrue~=uClass(c));
                FPc = sum(ypred_lbl==uClass(c) & ytrue~=uClass(c)); 
                FNc = sum(ypred_lbl~=uClass(c) & ytrue==uClass(c));
                SA(c)=TPc/(TPc+FNc+eps); 
                SpA(c)=TNc/(TNc+FPc+eps); 
                F1A(c)=2*TPc/(2*TPc+FPc+FNc+eps);
                MCCA(c)=(TPc*TNc-FPc*FNc)/sqrt((TPc+FPc)*(TPc+FNc)*(TNc+FPc)*(TNc+FNc)+eps);
                try 
                    [~,~,~,aucc] = perfcurve(ytrue, prob(:,c), uClass(c)); 
                    AUCA(c)=aucc; 
                catch
                    AUCA(c)=NaN; 
                end
            end
            Sens=mean(SA,'omitnan'); 
            Spec=mean(SpA,'omitnan'); 
            F1=mean(F1A,'omitnan'); 
            MCC=mean(MCCA,'omitnan'); 
            AUC=mean(AUCA,'omitnan'); 
            Brier=mean(sum((prob-yoh).^2,2),'omitnan'); 
            thr=NaN;
        catch
            [Acc,AUC,Sens,Spec,F1,MCC,Brier,thr] = deal(NaN); 
        end
    end
    res = [Acc, AUC, Sens, Spec, F1, MCC, Brier, thr];
end

function visualizeOutputs(models, numC, uClass, ML_Labels_Int, ML_Probs_Int, ML_Preds_Int, ML_Labels_Ext, ML_Probs_Ext, ML_Preds_Ext, meanInt, stdInt, meanExt, stdExt, intRes, extRes)
    nModels = numel(models);
    
    mI = squeeze(meanInt);  
    sI = squeeze(stdInt);   
    mE = squeeze(meanExt);  
    sE = squeeze(stdExt);   
    if nModels == 1
        mI = mI(:)'; sI = sI(:)'; mE = mE(:)'; sE = sE(:)';
    end

    for m = 1:nModels
        figure('Name', sprintf('Detailed Validation Analysis: %s', models{m}), 'Color', 'w');
        tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
        
        nIter = size(intRes, 2);
        N_int = floor(length(ML_Labels_Int{m}) / nIter);
        N_ext = floor(length(ML_Labels_Ext{m}) / nIter);
        
        fixed_FPR = linspace(0, 1, 100);
        sum_tpr_pos_i = zeros(1, 100); sum_tpr_neg_i = zeros(1, 100);
        sum_tpr_pos_e = zeros(1, 100); sum_tpr_neg_e = zeros(1, 100);
        sum_tpr_mc_i  = zeros(numC, 100); sum_tpr_mc_e = zeros(numC, 100);
        mean_cm_int = zeros(numC, numC);
        mean_cm_ext = zeros(numC, numC);
        uClass_sorted = sort(uClass);
        
        for iter_idx = 1:nIter
            idx_i = (iter_idx-1)*N_int+1 : iter_idx*N_int;
            idx_e = (iter_idx-1)*N_ext+1 : iter_idx*N_ext;
            
            iter_Y_int   = ML_Labels_Int{m}(idx_i);
            iter_P_int   = ML_Probs_Int{m}(idx_i, :);
            iter_Pred_int = ML_Preds_Int{m}(idx_i);
            iter_Y_ext   = ML_Labels_Ext{m}(idx_e);
            iter_P_ext   = ML_Probs_Ext{m}(idx_e, :);
            iter_Pred_ext = ML_Preds_Ext{m}(idx_e);
            
            mean_cm_int = mean_cm_int + confusionmat(iter_Y_int, iter_Pred_int, 'Order', uClass_sorted);
            mean_cm_ext = mean_cm_ext + confusionmat(iter_Y_ext, iter_Pred_ext, 'Order', uClass_sorted);
            
            if numC == 2
                posC = max(uClass); negC = min(uClass);
                [fpr,tpr,~,~] = perfcurve(iter_Y_int, iter_P_int(:,1), posC);
                [u_fpr,uid] = unique(fpr, 'last');
                sum_tpr_pos_i = sum_tpr_pos_i + max(0, min(1, interp1(u_fpr, tpr(uid), fixed_FPR, 'linear', 0)));
                
                [fpr,tpr,~,~] = perfcurve(iter_Y_int, 1-iter_P_int(:,1), negC);
                [u_fpr,uid] = unique(fpr, 'last');
                sum_tpr_neg_i = sum_tpr_neg_i + max(0, min(1, interp1(u_fpr, tpr(uid), fixed_FPR, 'linear', 0)));
                
                [fpr,tpr,~,~] = perfcurve(iter_Y_ext, iter_P_ext(:,1), posC);
                [u_fpr,uid] = unique(fpr, 'last');
                sum_tpr_pos_e = sum_tpr_pos_e + max(0, min(1, interp1(u_fpr, tpr(uid), fixed_FPR, 'linear', 0)));
                
                [fpr,tpr,~,~] = perfcurve(iter_Y_ext, 1-iter_P_ext(:,1), negC);
                [u_fpr,uid] = unique(fpr, 'last');
                sum_tpr_neg_e = sum_tpr_neg_e + max(0, min(1, interp1(u_fpr, tpr(uid), fixed_FPR, 'linear', 0)));
            else
                for c = 1:numC
                    [fpr,tpr,~,~] = perfcurve(iter_Y_int, iter_P_int(:,c), uClass(c));
                    [u_fpr,uid] = unique(fpr, 'last');
                    sum_tpr_mc_i(c,:) = sum_tpr_mc_i(c,:) + max(0, min(1, interp1(u_fpr, tpr(uid), fixed_FPR, 'linear', 0)));
                    
                    [fpr,tpr,~,~] = perfcurve(iter_Y_ext, iter_P_ext(:,c), uClass(c));
                    [u_fpr,uid] = unique(fpr, 'last');
                    sum_tpr_mc_e(c,:) = sum_tpr_mc_e(c,:) + max(0, min(1, interp1(u_fpr, tpr(uid), fixed_FPR, 'linear', 0)));
                end
            end
        end
        
        mean_cm_int = mean_cm_int / nIter;
        mean_cm_ext = mean_cm_ext / nIter;
        avg_tpr_pos_i = sum_tpr_pos_i / nIter; avg_tpr_neg_i = sum_tpr_neg_i / nIter;
        avg_tpr_pos_e = sum_tpr_pos_e / nIter; avg_tpr_neg_e = sum_tpr_neg_e / nIter;
        avg_tpr_mc_i  = sum_tpr_mc_i / nIter;  avg_tpr_mc_e  = sum_tpr_mc_e / nIter;
        
        % TILE 1: Internal ROC
        nexttile; hold on; grid on;
        mAUC_I = mI(m,2)*100; sAUC_I = sI(m,2)*100;
        if numC == 2
            posC = max(uClass); negC = min(uClass);
            plot([0, fixed_FPR], [0, avg_tpr_pos_i], 'r', 'LineWidth', 2, 'DisplayName', sprintf('Positive Class (%d)', posC));
            [~,ip] = max(avg_tpr_pos_i - fixed_FPR);
            plot(fixed_FPR(ip), avg_tpr_pos_i(ip), 'rs', 'MarkerSize', 8, 'LineWidth', 2, 'DisplayName', 'Op.Point (Pos)');
            plot([0, fixed_FPR], [0, avg_tpr_neg_i], 'b', 'LineWidth', 2, 'DisplayName', sprintf('Negative Class (%d)', negC));
            [~,in_] = max(avg_tpr_neg_i - fixed_FPR);
            plot(fixed_FPR(in_), avg_tpr_neg_i(in_), 'bs', 'MarkerSize', 8, 'LineWidth', 2, 'DisplayName', 'Op.Point (Neg)');
        else
            for c = 1:numC
                plot([0, fixed_FPR], [0, avg_tpr_mc_i(c,:)], 'LineWidth', 2, 'DisplayName', sprintf('Class %d', uClass(c)));
            end
        end
        plot([0 1],[0 1],'k--','DisplayName','Random');
        title(sprintf('%s - Internal ROC (AUC = %.1f%% ± %.1f%%)', models{m}, mAUC_I, sAUC_I), 'FontWeight','bold');
        xlabel('False Positive Rate'); ylabel('True Positive Rate'); legend('Location','southeast');
        
        % TILE 2: External ROC
        nexttile; hold on; grid on;
        mAUC_E = mE(m,2)*100; sAUC_E = sE(m,2)*100;
        if numC == 2
            posC = max(uClass); negC = min(uClass);
            plot([0, fixed_FPR], [0, avg_tpr_pos_e], 'r', 'LineWidth', 2, 'DisplayName', sprintf('Positive Class (%d)', posC));
            [~,ip] = max(avg_tpr_pos_e - fixed_FPR);
            plot(fixed_FPR(ip), avg_tpr_pos_e(ip), 'rs', 'MarkerSize', 8, 'LineWidth', 2, 'DisplayName', 'Op.Point (Pos)');
            plot([0, fixed_FPR], [0, avg_tpr_neg_e], 'b', 'LineWidth', 2, 'DisplayName', sprintf('Negative Class (%d)', negC));
            [~,in_] = max(avg_tpr_neg_e - fixed_FPR);
            plot(fixed_FPR(in_), avg_tpr_neg_e(in_), 'bs', 'MarkerSize', 8, 'LineWidth', 2, 'DisplayName', 'Op.Point (Neg)');
        else
            for c = 1:numC
                plot([0, fixed_FPR], [0, avg_tpr_mc_e(c,:)], 'LineWidth', 2, 'DisplayName', sprintf('Class %d', uClass(c)));
            end
        end
        plot([0 1],[0 1],'k--','DisplayName','Random');
        title(sprintf('%s - External ROC (AUC = %.1f%% ± %.1f%%)', models{m}, mAUC_E, sAUC_E), 'FontWeight','bold');
        xlabel('False Positive Rate'); ylabel('True Positive Rate'); legend('Location','southeast');
        
        % TILE 3: AUC Boxplot + Wilcoxon
        nexttile;
        auc_int = squeeze(intRes(m,:,2)) * 100; auc_int = auc_int(:);
        auc_ext = squeeze(extRes(m,:,2)) * 100; auc_ext = auc_ext(:);
        if isscalar(auc_int), auc_int = [auc_int; auc_int]; auc_ext = [auc_ext; auc_ext]; end
        warning('off','all');
        boxplot([auc_int, auc_ext], 'Labels', {'Internal CV','External Holdout'});
        warning('on','all');
        try
            p_auc = signrank(auc_int(:), auc_ext(:));
            p_str = sprintf('Wilcoxon p=%.4f%s', p_auc, repmat(' (*)',1,p_auc<0.05));
        catch
            p_str = 'Wilcoxon p=N/A';
        end
        title(sprintf('%s - AUC Distribution (%s)', models{m}, p_str), 'FontWeight','bold');
        ylabel('AUC (%)'); grid on;
                
        % TILE 4: Internal CM (mean ± std)
        nexttile;
        cm_int = confusionchart(round(fillmissing(mean_cm_int, 'constant', 0)), string(uClass_sorted));
        cm_int.Title = sprintf('%s - Internal CM (Acc: %.1f%% ± %.2f%%)', models{m}, mI(m,1)*100, sI(m,1)*100);
        cm_int.DiagonalColor = [0.15 0.25 0.8];
        cm_int.OffDiagonalColor = [0.8 0.2 0.2];
        
        % TILE 5: External CM (mean ± std)
        nexttile;
        cm_ext = confusionchart(round(fillmissing(mean_cm_ext, 'constant', 0)), string(uClass_sorted));
        cm_ext.Title = sprintf('%s - External CM (Acc: %.1f%% ± %.2f%%)', models{m}, mE(m,1)*100, sE(m,1)*100);
        cm_ext.DiagonalColor = [0.15 0.25 0.8];
        cm_ext.OffDiagonalColor = [0.8 0.2 0.2];
        
        % TILE 6: ACC Boxplot + Wilcoxon
        nexttile;
        acc_int = squeeze(intRes(m,:,1)) * 100; acc_int = acc_int(:);
        acc_ext = squeeze(extRes(m,:,1)) * 100; acc_ext = acc_ext(:);
        if isscalar(acc_int), acc_int = [acc_int; acc_int]; acc_ext = [acc_ext; acc_ext]; end
        warning('off','all');
        boxplot([acc_int, acc_ext], 'Labels', {'Internal CV','External Holdout'});
        warning('on','all');
        try
            p_acc = signrank(acc_int(:), acc_ext(:));
            p_str = sprintf('Wilcoxon p=%.4f%s', p_acc, repmat(' (*)',1,p_acc<0.05));
        catch
            p_str = 'Wilcoxon p=N/A';
        end
        title(sprintf('%s - ACC Distribution (%s)', models{m}, p_str), 'FontWeight','bold');
        ylabel('Accuracy (%)'); grid on;
        
        sgtitle(sprintf('Validation Analysis: %s', models{m}), 'FontWeight','bold','FontSize',13);
    end
end
