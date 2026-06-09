classdef ML_Auto_GUI < handle
    properties (Access = private)
        UIFigure      matlab.ui.Figure
        MainGrid      matlab.ui.container.GridLayout
        LeftPanel     matlab.ui.container.Panel
        RightPanel    matlab.ui.container.TabGroup
        LogArea       matlab.ui.control.TextArea
        LoadBtn       matlab.ui.control.Button
        RunBtn        matlab.ui.control.Button
        ExpBtn        matlab.ui.control.Button
        ShapBtn       matlab.ui.control.Button
        CalibBtn      matlab.ui.control.Button
        ModelDrop     matlab.ui.control.DropDown
        TopNDrop      matlab.ui.control.DropDown
        CalibDrop     matlab.ui.control.DropDown
        PerfDrop      matlab.ui.control.DropDown
        PerfBtn       matlab.ui.control.Button
        KEdit         matlab.ui.control.NumericEditField
        IterEdit      matlab.ui.control.NumericEditField
        HoldEdit      matlab.ui.control.NumericEditField
        ModelChecks   struct
        LoadedData    table
        Results       struct
    end

    methods
        function obj = ML_Auto_GUI()
            obj.createUI();
        end
    end

    methods (Access = private)
        function createUI(obj)
            obj.UIFigure = uifigure('Name', 'AutoML Pro - Academic Analysis Panel', 'Color', 'w', 'Position', [100 100 1050 650]);
            obj.MainGrid = uigridlayout(obj.UIFigure, [1 2], 'ColumnWidth', {290, '1x'}, 'RowHeight', {'1x'}, 'Padding', [10 10 10 10]);
            obj.LeftPanel = uipanel(obj.MainGrid, 'Title', 'CONFIGURATION', 'BackgroundColor', '#f8f9fa', 'Scrollable', 'on');
            
            uilabel(obj.LeftPanel, 'Position', [20 600 200 22], 'Text', '1. Data Source:', 'FontWeight', 'bold');
            obj.LoadBtn = uibutton(obj.LeftPanel, 'Position', [20 570 240 30], 'Text', 'Load Dataset (.csv, .xlsx, .mat)', 'ButtonPushedFcn', @(btn, event) obj.loadData());

            uilabel(obj.LeftPanel, 'Position', [20 545 200 22], 'Text', '2. Analysis Parameters:', 'FontWeight', 'bold');
            uilabel(obj.LeftPanel, 'Position', [20 520 100 22], 'Text', 'K-Fold:');
            obj.KEdit = uieditfield(obj.LeftPanel, 'numeric', 'Position', [130 520 130 22], 'Value', 10);
            uilabel(obj.LeftPanel, 'Position', [20 495 100 22], 'Text', 'Iterations:');
            obj.IterEdit = uieditfield(obj.LeftPanel, 'numeric', 'Position', [130 495 130 22], 'Value', 10);
            uilabel(obj.LeftPanel, 'Position', [20 470 100 22], 'Text', 'Holdout (%):');
            obj.HoldEdit = uieditfield(obj.LeftPanel, 'numeric', 'Position', [130 470 130 22], 'Value', 10);

            uilabel(obj.LeftPanel, 'Position', [20 445 200 22], 'Text', '3. Model Selection:', 'FontWeight', 'bold');
            uilabel(obj.LeftPanel, 'Position', [20 425 120 22], 'Text', 'White-Box', 'FontWeight', 'bold', 'FontColor', [0.3 0.3 0.3]);
            uilabel(obj.LeftPanel, 'Position', [150 425 130 22], 'Text', 'Black-Box', 'FontWeight', 'bold', 'FontColor', [0.3 0.3 0.3]);

            obj.ModelChecks = struct(); 
            mListWhite = {'DT', 'LR', 'DA', 'NB', 'GAM', 'GLM'};
            for i = 1:numel(mListWhite), obj.ModelChecks.(mListWhite{i}) = uicheckbox(obj.LeftPanel, 'Position', [20 405-(i-1)*22 110 22], 'Text', mListWhite{i}, 'Value', 0); end
            
            mListBlack = {'KNN', 'SVM', 'NN', 'ENS'};
            for i = 1:numel(mListBlack), obj.ModelChecks.(mListBlack{i}) = uicheckbox(obj.LeftPanel, 'Position', [150 405-(i-1)*22 110 22], 'Text', mListBlack{i}, 'Value', 0); end

            obj.RunBtn = uibutton(obj.LeftPanel, 'Position', [20 240 240 40], 'Text', 'START ANALYSIS', 'FontWeight', 'bold', 'BackgroundColor', '#e67e22', 'FontColor', 'w', 'Enable', 'off', 'ButtonPushedFcn', @(btn, event) obj.runAnalysis());
            obj.ExpBtn = uibutton(obj.LeftPanel, 'Position', [20 205 240 30], 'Text', 'Export Results & Models', 'Enable', 'off', 'ButtonPushedFcn', @(btn, event) obj.exportResults());
            
            uilabel(obj.LeftPanel, 'Position', [20 175 200 22], 'Text', '4. SHAP Analysis:', 'FontWeight', 'bold');
            obj.ModelDrop = uidropdown(obj.LeftPanel, 'Position', [20 150 155 25], 'Items', {'Pending Analysis...'}, 'Enable', 'off');
            obj.ShapBtn = uibutton(obj.LeftPanel, 'Position', [185 150 75 25], 'Text', 'Plot', 'BackgroundColor', '#2ecc71', 'FontWeight', 'bold', 'Enable', 'off', 'ButtonPushedFcn', @(btn, event) obj.plot_SHAP(obj.ModelDrop.Value));
            uilabel(obj.LeftPanel, 'Position', [20 120 165 25], 'Text', 'Top N Features:');
            obj.TopNDrop = uidropdown(obj.LeftPanel, 'Position', [185 120 75 25], 'Items', {'-'}, 'Enable', 'off');

            uilabel(obj.LeftPanel, 'Position', [20 90 200 22], 'Text', '5. Calibration Curve:', 'FontWeight', 'bold');
            obj.CalibDrop = uidropdown(obj.LeftPanel, 'Position', [20 65 155 25], 'Items', {'Pending Analysis...'}, 'Enable', 'off');
            obj.CalibBtn = uibutton(obj.LeftPanel, 'Position', [185 65 75 25], 'Text', 'Plot', 'BackgroundColor', '#3498db', 'FontColor', 'w', 'FontWeight', 'bold', 'Enable', 'off', 'ButtonPushedFcn', @(btn, event) obj.plot_Calibration(obj.CalibDrop.Value));
            
            uilabel(obj.LeftPanel, 'Position', [20 35 200 22], 'Text', '6. Performance Plots:', 'FontWeight', 'bold');
            obj.PerfDrop = uidropdown(obj.LeftPanel, 'Position', [20 10 155 25], 'Items', {'Pending Analysis...'}, 'Enable', 'off');
            obj.PerfBtn = uibutton(obj.LeftPanel, 'Position', [185 10 75 25], 'Text', 'Plot', 'BackgroundColor', '#9b59b6', 'FontColor', 'w', 'FontWeight', 'bold', 'Enable', 'off', 'ButtonPushedFcn', @(btn, event) obj.plot_Performance(obj.PerfDrop.Value));
            
            obj.RightPanel = uitabgroup(obj.MainGrid);
            logTab = uitab(obj.RightPanel, 'Title', 'Analysis Log');
            logGrid = uigridlayout(logTab, [1 1], 'Padding', [5 5 5 5]);
            obj.LogArea = uitextarea(logGrid, 'Editable', 'off', 'BackgroundColor', 'k', 'FontColor', 'g', 'FontName', 'Consolas');
            obj.addLog('System ready. Please load a dataset.');
        end

        function addLog(obj, txt)
            if isvalid(obj.LogArea)
                timeStr = char(datetime('now', 'Format', 'HH:mm:ss'));
                obj.LogArea.Value = [obj.LogArea.Value; {sprintf('[%s] %s', timeStr, txt)}];
                scroll(obj.LogArea, 'bottom'); drawnow;
            end
        end

        function loadData(obj)
            [file, fpath] = uigetfile({'*.csv;*.xlsx;*.xls;*.mat'});
            if isequal(file, 0), return; end
            fullPath = fullfile(fpath, file); [~, ~, ext] = fileparts(file);
            try
                if strcmpi(ext, '.mat')
                    tmp = load(fullPath); v = fieldnames(tmp); isValid = false;
                    for k = 1:numel(v)
                        var = tmp.(v{k});
                        if istable(var) || (isnumeric(var) && ismatrix(var))
                            if isnumeric(var), obj.LoadedData = array2table(var); else, obj.LoadedData = var; end
                            isValid = true; break;
                        end
                    end
                    if ~isValid, uialert(obj.UIFigure, 'ERROR: Valid data not found in .mat file.', 'Format Error'); return; end
                else, obj.LoadedData = readtable(fullPath, 'VariableNamingRule', 'preserve'); end
                nFeat = size(obj.LoadedData, 2) - 1;
                strItems = arrayfun(@num2str, 1:nFeat, 'UniformOutput', false);
                obj.TopNDrop.Items = strItems; obj.TopNDrop.Value = strItems{min(15, nFeat)}; obj.TopNDrop.Enable = 'on';
                obj.addLog(['Dataset loaded: ' file ' (' num2str(nFeat) ' Features Detected)']); obj.RunBtn.Enable = 'on';
            catch ME, uialert(obj.UIFigure, ['Data reading error: ' ME.message], 'Error'); end
        end

        function runAnalysis(obj)
            mNames = fields(obj.ModelChecks); activeModels = {};
            for i = 1:numel(mNames), if obj.ModelChecks.(mNames{i}).Value, activeModels{end+1} = mNames{i}; end, end
            if isempty(activeModels), uialert(obj.UIFigure, 'Select at least one model!', 'Warning'); return; end
            
            % Detect the number of classes (numC) from the target variable
            if istable(obj.LoadedData)
                y_temp = table2array(obj.LoadedData(:, end));
            else
                y_temp = obj.LoadedData(:, end);
            end
            
            numC = numel(unique(y_temp(~isnan(y_temp))));
            
            % Filter out GAM if the dataset is multiclass due to MATLAB API limitation
            if numC > 2 && ismember('GAM', activeModels)
                activeModels = setdiff(activeModels, {'GAM'}, 'stable');
                obj.addLog('MATLAB API Limitation: GAM excluded from multiclass analysis due to lack of native ECOC support.');
            end

            obj.RunBtn.Enable = 'off'; obj.addLog('Analysis started...');
            set(obj.UIFigure, 'Pointer', 'watch'); drawnow; 
            try
                [IntT, ExtT, TimT, OptT, StatT, OptRaw, ML_Probs_Int, ML_Labels_Int, ML_Probs_Ext, ML_Labels_Ext, extRes, timeRes, BestM, PlotData, PlotFunc] = ...
                    ML_Repeated_Nested_CV(obj.LoadedData, obj.KEdit.Value, obj.HoldEdit.Value, obj.IterEdit.Value, activeModels, @(t) obj.addLog(t));
                
                obj.Results = struct();
                obj.Results.InternalTable = IntT;
                obj.Results.ExternalTable = ExtT;
                obj.Results.TimeTable = TimT;
                obj.Results.StatTable = StatT;
                obj.Results.OptTable = OptT;
                obj.Results.BestModels = BestM;
                obj.Results.ActiveModels = activeModels;
                obj.Results.ML_Probs_Int = ML_Probs_Int;
                obj.Results.ML_Labels_Int = ML_Labels_Int;
                obj.Results.ML_Probs_Ext = ML_Probs_Ext;
                obj.Results.ML_Labels_Ext = ML_Labels_Ext;
                obj.Results.extRes = extRes;
                obj.Results.timeRes = timeRes;
                obj.Results.OptimizedParamsRaw = OptRaw;
                obj.Results.PlotData = PlotData;
                obj.Results.PlotFunc = PlotFunc;
                
                obj.createResultTabs(IntT, ExtT, TimT, StatT, OptT); assignin('base', 'BestModels', BestM);
                obj.ModelDrop.Items = activeModels; obj.ModelDrop.Enable = 'on'; obj.ShapBtn.Enable = 'on';
                obj.CalibDrop.Items = activeModels; obj.CalibDrop.Enable = 'on'; obj.CalibBtn.Enable = 'on';
                obj.PerfDrop.Items = activeModels; obj.PerfDrop.Enable = 'on'; obj.PerfBtn.Enable = 'on';
                obj.ExpBtn.Enable = 'on'; obj.RunBtn.Enable = 'on'; obj.addLog('Analysis complete. BestModels exported to Workspace.');
            catch ME
                obj.addLog(['ERROR: ' ME.message]); 
                obj.addLog(getReport(ME, 'extended', 'hyperlinks', 'off'));
                obj.RunBtn.Enable = 'on'; 
            end
            set(obj.UIFigure, 'Pointer', 'arrow'); drawnow; 
        end

        function createResultTabs(obj, IntT, ExtT, TimT, StatT, OptT)
            allTabs = obj.RightPanel.Children;
            for i = 1:numel(allTabs), if ~strcmp(allTabs(i).Title, 'Analysis Log'), delete(allTabs(i)); end, end
            t1 = uitab(obj.RightPanel, 'Title', 'Internal CV'); g1 = uigridlayout(t1, [1 1], 'Padding', [0 0 0 0]); uitable(g1, 'Data', IntT);
            t2 = uitab(obj.RightPanel, 'Title', 'External Holdout'); g2 = uigridlayout(t2, [1 1], 'Padding', [0 0 0 0]); uitable(g2, 'Data', ExtT);
            t3 = uitab(obj.RightPanel, 'Title', 'Time Analysis'); g3 = uigridlayout(t3, [1 1], 'Padding', [0 0 0 0]); uitable(g3, 'Data', TimT);
            t4 = uitab(obj.RightPanel, 'Title', 'Wilcoxon Test'); g4 = uigridlayout(t4, [2 1], 'RowHeight', {'1x', 30}, 'Padding', [5 5 5 5]); 
            uitable(g4, 'Data', StatT, 'RowName', StatT.Properties.RowNames);
            uilabel(g4, 'Text', 'NOTE: Values with p < 0.05 (*) indicate statistical significance.', 'FontWeight', 'bold', 'FontColor', 'r');
            t5 = uitab(obj.RightPanel, 'Title', 'Optimal Parameters'); g5 = uigridlayout(t5, [1 1], 'Padding', [0 0 0 0]); uitable(g5, 'Data', OptT);
            obj.RightPanel.SelectedTab = t1;
        end

        function exportResults(obj)
            [file, fpath] = uiputfile('AutoML_Analysis_Results.xlsx'); if isequal(file, 0), return; end
            p = fullfile(fpath, file); 
            
            writetable(obj.Results.InternalTable, p, 'Sheet', 'Internal');
            writetable(obj.Results.ExternalTable, p, 'Sheet', 'External'); 
            writetable(obj.Results.TimeTable, p, 'Sheet', 'Time_Analysis');
            writetable(obj.Results.StatTable, p, 'Sheet', 'Statistical_Test', 'WriteRowNames', true);
            writetable(obj.Results.OptTable, p, 'Sheet', 'Optimal_Parameters'); 
            
            matFile = strrep(p, '.xlsx', '_AllResults.mat'); 
            AllResults = obj.Results; 
            save(matFile, 'AllResults');
            
            obj.addLog(sprintf('Excel saved: %s', p));
            obj.addLog(sprintf('All raw data (Struct) saved: %s', matFile));
        end
        
        function plot_Calibration(obj, mName)
            if strcmp(mName, 'Pending Analysis...') || isempty(mName), uialert(obj.UIFigure, 'Select a model.', 'Warning'); return; end
            set(obj.UIFigure, 'Pointer', 'watch'); drawnow;
            try
                rng(42); 
                mdl = obj.Results.BestModels.(mName); 
                mIdx = find(strcmp(obj.Results.ActiveModels, mName));
                
                Y_int = obj.Results.ML_Labels_Int{mIdx};
                s_int = obj.Results.ML_Probs_Int{mIdx};
                Y_ext = obj.Results.ML_Labels_Ext{mIdx};
                s_ext = obj.Results.ML_Probs_Ext{mIdx};
                
                if isprop(mdl, 'ClassNames'), classes = mdl.ClassNames; else, classes = unique([Y_int; Y_ext]); end
                numClasses = numel(classes); 
                nIter = size(obj.Results.extRes, 2); 
                
                N_int = floor(length(Y_int) / nIter);
                N_ext = floor(length(Y_ext) / nIter);
                
                iterClasses = 1:numClasses; if numClasses == 2, iterClasses = 2; end
                
                fCal = figure('Name', ['Calibration: ' mName], 'Color', 'w', 'Position', [100 100 1100 550]); movegui(fCal, 'center');
                colors = lines(numClasses); tblData = cell(0, 6); edges = linspace(0, 1, 11);
                
                ax1 = subplot(1,2,1); hold(ax1, 'on'); grid(ax1, 'on');
                plot(ax1, [0 1], [0 1], 'k--', 'LineWidth', 1.5, 'DisplayName', 'Ideal');
                ax2 = subplot(1,2,2); hold(ax2, 'on'); grid(ax2, 'on');
                plot(ax2, [0 1], [0 1], 'k--', 'LineWidth', 1.5, 'DisplayName', 'Ideal');
                
                for c = iterClasses
                    if size(s_int, 2) == 1
                        p_int = s_int; p_ext = s_ext; 
                    else
                        p_int = s_int(:, c); p_ext = s_ext(:, c); 
                    end
                    
                    iter_tF_i = nan(nIter, 10); iter_mP_i = nan(nIter, 10);
                    iter_tF_e = nan(nIter, 10); iter_mP_e = nan(nIter, 10);
                    
                    for i = 1:nIter
                        idx_i = (i - 1) * N_int + 1 : i * N_int;
                        idx_e = (i - 1) * N_ext + 1 : i * N_ext;
                        
                        iter_Y_i = Y_int(idx_i); iter_P_i = p_int(idx_i);
                        iter_Y_e = Y_ext(idx_e); iter_P_e = p_ext(idx_e);
                        
                        t_int_iter = (iter_Y_i == classes(c));
                        t_ext_iter = (iter_Y_e == classes(c));
                        
                        for b = 1:10
                            if b==10, in_i = (iter_P_i>=edges(b) & iter_P_i<=edges(b+1)); in_e = (iter_P_e>=edges(b) & iter_P_e<=edges(b+1));
                            else, in_i = (iter_P_i>=edges(b) & iter_P_i<edges(b+1)); in_e = (iter_P_e>=edges(b) & iter_P_e<edges(b+1)); end
                            
                            if sum(in_i) > 0
                                iter_mP_i(i, b) = mean(iter_P_i(in_i));
                                iter_tF_i(i, b) = sum(t_int_iter(in_i)) / sum(in_i);
                            end
                            if sum(in_e) > 0
                                iter_mP_e(i, b) = mean(iter_P_e(in_e));
                                iter_tF_e(i, b) = sum(t_ext_iter(in_e)) / sum(in_e);
                            end
                        end
                    end
                    
                    mean_P_i = mean(iter_mP_i, 1, 'omitnan'); mean_F_i = mean(iter_tF_i, 1, 'omitnan'); std_F_i = std(iter_tF_i, 0, 1, 'omitnan');
                    mean_P_e = mean(iter_mP_e, 1, 'omitnan'); mean_F_e = mean(iter_tF_e, 1, 'omitnan'); std_F_e = std(iter_tF_e, 0, 1, 'omitnan');
                    
                    for b = 1:10
                        rngStr = sprintf('%.1f-%.1f', edges(b), edges(b+1));
                        val_pi = '-'; val_ti = '-'; val_pe = '-'; val_te = '-';
                        if ~isnan(mean_P_i(b)), val_pi = sprintf('%.4f', mean_P_i(b)); val_ti = sprintf('%.4f', mean_F_i(b)); end
                        if ~isnan(mean_P_e(b)), val_pe = sprintf('%.4f', mean_P_e(b)); val_te = sprintf('%.4f', mean_F_e(b)); end
                        if ~isnan(mean_P_i(b)) || ~isnan(mean_P_e(b))
                            tblData(end+1,:) = {string(classes(c)), rngStr, val_pi, val_ti, val_pe, val_te};
                        end
                    end
                    
                    v_i = ~isnan(mean_P_i) & ~isnan(mean_F_i); 
                    v_e = ~isnan(mean_P_e) & ~isnan(mean_F_e);
                    
                    if numClasses == 2, nm = 'Target Class'; else, nm = sprintf('Class %s', string(classes(c))); end
                    
                    plot(ax1, mean_P_i(v_i), mean_F_i(v_i), '-o', 'LineWidth', 2, 'MarkerSize', 6, 'MarkerFaceColor', colors(c,:), 'Color', colors(c,:), 'DisplayName', nm);
                    plot(ax2, mean_P_e(v_e), mean_F_e(v_e), '-s', 'LineWidth', 2, 'MarkerSize', 6, 'MarkerFaceColor', 'w', 'Color', colors(c,:), 'DisplayName', nm);
                end
                
                title(ax1, 'Internal CV'); xlabel(ax1, 'Mean Predicted Prob.'); ylabel(ax1, 'Fraction of Positives'); legend(ax1, 'Location', 'southeast', 'FontSize', 8);
                title(ax2, 'External Holdout'); xlabel(ax2, 'Mean Predicted Prob.'); ylabel(ax2, 'Fraction of Positives'); legend(ax2, 'Location', 'southeast', 'FontSize', 8);
                sgtitle(['Calibration Analysis (Reliability Diagrams): ' mName], 'FontWeight', 'bold');
                
                tabTitle = ['Calibration Table (' mName ')']; allTabs = obj.RightPanel.Children;
                for i=1:numel(allTabs), if strcmp(allTabs(i).Title, tabTitle), delete(allTabs(i)); end; end 
                
                tCalTab = uitab(obj.RightPanel, 'Title', tabTitle); 
                gC = uigridlayout(tCalTab, [2 1], 'RowHeight', {30, '1x'}, 'Padding', [5 5 5 5]);
                
                closeBtn = uibutton(gC, 'Text', 'Close Tab', 'BackgroundColor', '#e74c3c', 'FontColor', 'w', 'FontWeight', 'bold');
                closeBtn.ButtonPushedFcn = @(btn, event) delete(tCalTab);
                
                calTbl = cell2table(tblData, 'VariableNames', {'Class', 'Range', 'Int_Pred', 'Int_Actual', 'Ext_Pred', 'Ext_Actual'});
                uitable(gC, 'Data', calTbl); 
                obj.RightPanel.SelectedTab = tCalTab;
                
                obj.addLog(sprintf('Calibration generated for %s.', mName));
            catch ME, obj.addLog(['Error: ' ME.message]); end
            set(obj.UIFigure, 'Pointer', 'arrow'); drawnow;
        end
        
        function plot_SHAP(obj, mName)
            if strcmp(mName, 'Pending Analysis...') || isempty(mName), uialert(obj.UIFigure, 'Select a model.', 'Warning'); return; end
            set(obj.UIFigure, 'Pointer', 'watch'); drawnow; 
            try
                rng(42); 
                if istable(obj.LoadedData), Xraw = table2array(obj.LoadedData(:, 1:end-1)); fNamesRaw = obj.LoadedData.Properties.VariableNames(1:end-1);
                else, Xraw = obj.LoadedData(:, 1:end-1); fNamesRaw = arrayfun(@(x) ['F' num2str(x)], 1:size(Xraw,2), 'UniformOutput', 0); end
                
                mdl = obj.Results.BestModels.(mName); 
                prepName = [mName '_Preprocess'];
                testDataName = [mName '_TestX'];
                trainDataName = [mName '_TrainX'];
                
                if isfield(obj.Results.BestModels, testDataName)
                    X_test = obj.Results.BestModels.(testDataName);
                    if isfield(obj.Results.BestModels, prepName)
                        prep = obj.Results.BestModels.(prepName); fNames = fNamesRaw(prep.keepIdx);
                    else, fNames = fNamesRaw; end
                else
                    if isfield(obj.Results.BestModels, prepName)
                        prep = obj.Results.BestModels.(prepName); X_test = Xraw(:, prep.keepIdx); fNames = fNamesRaw(prep.keepIdx);
                    else, X_test = Xraw; fNames = fNamesRaw; end
                end

                if isfield(obj.Results.BestModels, trainDataName)
                    X_train = obj.Results.BestModels.(trainDataName);
                    if size(X_train, 1) > 100
                        idx = randperm(size(X_train, 1), 100);
                        X_train = X_train(idx, :);
                    end
                else
                    X_train = X_test; 
                end

                if isprop(mdl, 'ClassNames')
                    classes = mdl.ClassNames; numClasses = numel(classes); 
                elseif isa(mdl, 'GeneralizedLinearModel')
                    mIdx = find(strcmp(obj.Results.ActiveModels, mName));
                    classes = unique(obj.Results.ML_Labels_Int{mIdx}); 
                    numClasses = numel(classes);
                else
                    numClasses = 1; 
                end
                
                if numClasses == 2, iterClasses = 2; nCols = 1; else, iterClasses = 1:numClasses; nCols = numClasses; end
                
                nFeat = length(fNames); 
                posImpact = zeros(nFeat, nCols); 
                negImpact = zeros(nFeat, nCols); 
                stdImpact = zeros(nFeat, nCols); 
                classNamesStr = cell(1, nCols);
                
                obj.addLog(sprintf('[%s] Computing Marginal Contribution SHAP on %d holdout samples...', mName, size(X_test,1))); drawnow;
                dlg = uiprogressdlg(obj.UIFigure, ...
                    'Title', sprintf('SHAP: %s', mName), ...
                    'Message', 'Initializing SHAP computation...', ...
                    'Indeterminate', 'on', ...
                    'Cancelable', 'off');
                drawnow;

                colIdx = 1;
                for c = iterClasses
                    if numClasses > 1
                        classNamesStr{colIdx} = sprintf('Class %s', string(classes(c))); 
                        scoreFcn = @(Xnew) obj.predict_prob(mdl, Xnew, c);
                        dlg.Message = sprintf('Computing SHAP for %s...\n(Kernel-based models may take several minutes)', classNamesStr{colIdx});
                        drawnow;
                        exp_c = shapley(scoreFcn, X_train); exp_c = fit(exp_c, X_test); raw = exp_c.ShapleyValues;
                    else
                        classNamesStr{colIdx} = 'Global Effect'; 
                        dlg.Message = sprintf('Computing SHAP...\n(Kernel-based models may take several minutes)');
                        drawnow;
                        exp_c = shapley(mdl, X_train); exp_c = fit(exp_c, X_test); raw = exp_c.ShapleyValues;
                    end
                    
                    if istable(raw), if ismember('QueryPoint', raw.Properties.VariableNames), raw.QueryPoint = []; end
                        numCols = varfun(@isnumeric, raw, 'OutputFormat', 'uniform'); sMat = table2array(raw(:, numCols));
                    else, sMat = double(raw); end
                    
                    if size(sMat, 2) == nFeat && size(sMat, 1) ~= nFeat, sMat = sMat'; end
                    if size(sMat, 1) > nFeat, sMat = sMat(1:nFeat, :); end
                    
                    for f = 1:nFeat
                        f_vals = sMat(f, :);
                        posImpact(f, colIdx) = sum(f_vals(f_vals > 0)) / length(f_vals);
                        negImpact(f, colIdx) = sum(f_vals(f_vals < 0)) / length(f_vals);
                        stdImpact(f, colIdx) = std(f_vals, 0, 2, 'omitnan');
                    end
                    colIdx = colIdx + 1;
                end
                
                totalImpact = sum(posImpact + abs(negImpact), 2); [~, descIdx] = sort(totalImpact, 'descend');
                topN = str2double(obj.TopNDrop.Value); if isnan(topN) || topN <= 0 || topN > nFeat, topN = nFeat; end
                tIdx = descIdx(1:topN); pIdx = flip(tIdx); 
                pS = posImpact(pIdx, :); nS = negImpact(pIdx, :); sS = stdImpact(pIdx, :); fN = fNames(pIdx);
                
                fShap = figure('Name', ['SHAP: ' mName], 'Color', 'w', 'Position', [100 100 1000 750]); movegui(fShap, 'center'); 
                ax = axes(fShap, 'Position', [0.30, 0.10, 0.65, 0.82]); hold(ax, 'on'); grid(ax, 'on');
                
                if nCols > 1
                    bPos = barh(ax, pS, 'grouped', 'EdgeColor', 'k');
                    bNeg = barh(ax, nS, 'grouped', 'EdgeColor', 'k');
                    colors = lines(nCols);
                    for c = 1:nCols
                        bPos(c).FaceColor = colors(c,:); bNeg(c).FaceColor = colors(c,:);
                        xp = bPos(c).YEndPoints; xn = bNeg(c).YEndPoints; yp = bPos(c).XEndPoints; stdc = sS(:, c)';
                        for i = 1:length(yp)
                            plot(ax, [xp(i), xp(i)+stdc(i)], [yp(i), yp(i)], 'k-', 'LineWidth', 1.0, 'HandleVisibility', 'off');
                            plot(ax, [xp(i)+stdc(i), xp(i)+stdc(i)], [yp(i)-0.15, yp(i)+0.15], 'k-', 'LineWidth', 1.0, 'HandleVisibility', 'off');
                            plot(ax, [xn(i), xn(i)-stdc(i)], [yp(i), yp(i)], 'k-', 'LineWidth', 1.0, 'HandleVisibility', 'off');
                            plot(ax, [xn(i)-stdc(i), xn(i)-stdc(i)], [yp(i)-0.15, yp(i)+0.15], 'k-', 'LineWidth', 1.0, 'HandleVisibility', 'off');
                        end
                    end
                    h_std = plot(ax, [nan nan], [nan nan], 'k-|', 'LineWidth', 1.5, 'MarkerSize', 8);
                    legend(ax, [bPos, h_std], [classNamesStr, {'SD Limit'}], 'Location', 'best');
                else
                    posClassName = string(classes(2)); 
                    negClassName = string(classes(1)); 
                    
                    for i = 1:length(pS)
                        barh(ax, i, pS(i), 'FaceColor', [0.85 0.32 0.09], 'EdgeColor', 'k', 'BarWidth', 0.6);
                        barh(ax, i, nS(i), 'FaceColor', [0 0.44 0.74], 'EdgeColor', 'k', 'BarWidth', 0.6);
                        
                        plot(ax, [pS(i), pS(i)+sS(i)], [i, i], 'k-', 'LineWidth', 1.5, 'HandleVisibility', 'off');
                        plot(ax, [pS(i)+sS(i), pS(i)+sS(i)], [i-0.2, i+0.2], 'k-', 'LineWidth', 1.5, 'HandleVisibility', 'off');
                        plot(ax, [nS(i), nS(i)-sS(i)], [i, i], 'k-', 'LineWidth', 1.5, 'HandleVisibility', 'off');
                        plot(ax, [nS(i)-sS(i), nS(i)-sS(i)], [i-0.2, i+0.2], 'k-', 'LineWidth', 1.5, 'HandleVisibility', 'off');
                    end
                    p1 = patch(ax, nan, nan, [0.85 0.32 0.09], 'EdgeColor', 'k'); p2 = patch(ax, nan, nan, [0 0.44 0.74], 'EdgeColor', 'k');
                    h_std = plot(ax, [nan nan], [nan nan], 'k-|', 'LineWidth', 1.5, 'MarkerSize', 8);
                    legend(ax, [p1, p2, h_std], {sprintf('Predicts Class %s (Positive Contrib.)', posClassName), sprintf('Predicts Class %s (Negative Contrib.)', negClassName), 'SD Limit'}, 'Location', 'best');
                end
                
                plot(ax, [0 0], [0 length(fN)+1], 'k-', 'LineWidth', 1.5, 'HandleVisibility', 'off'); 
                yticks(ax, 1:length(fN)); yticklabels(ax, strrep(fN, '_', ' ')); 
                xlabel(ax, 'Mean SHAP Value (Average Marginal Contribution to Class Probability \pm SD Limit)'); 
                title(ax, ['Global Non-Linear SHAP Explanations: ' mName], 'FontSize', 12, 'FontWeight', 'bold');
                
                sumText = cell(0,1); sumText{end+1} = sprintf('ACADEMIC SHAP SUMMARY (%s):', mName);
                for i = 1:length(tIdx)
                    fn = strrep(fNames{tIdx(i)}, '_', ' '); classDesc = {};
                    for c = 1:nCols
                        p_val = posImpact(tIdx(i), c); n_val = negImpact(tIdx(i), c);
                        if abs(p_val) >= 1e-4 || abs(n_val) >= 1e-4
                            if nCols == 1
                                classDesc{end+1} = sprintf('provides an average marginal contribution of %.3f towards predicting Class %s, and %.3f towards Class %s', p_val, posClassName, abs(n_val), negClassName);
                            else
                                cs = classNamesStr{c};
                                classDesc{end+1} = sprintf('for %s (Positive Contrib: %.3f | Negative Contrib: %.3f)', cs, p_val, abs(n_val));
                            end
                        end
                    end
                    if ~isempty(classDesc)
                        if length(classDesc) == 1, descStr = classDesc{1}; else, descStr = [strjoin(classDesc(1:end-1), ', '), ' | ', classDesc{end}]; end
                        sumText{end+1} = sprintf('- Feature [%s] %s.', fn, descStr);
                    end
                end
                
                tShapTab = uitab(obj.RightPanel, 'Title', ['SHAP Table (' mName ')']); 
                gS = uigridlayout(tShapTab, [3 1], 'RowHeight', {30, '1x', 150}, 'Padding', [5 5 5 5]);
                closeBtn = uibutton(gS, 'Text', 'Close Tab', 'BackgroundColor', '#e74c3c', 'FontColor', 'w', 'FontWeight', 'bold');
                closeBtn.ButtonPushedFcn = @(btn, event) delete(tShapTab);
                
                colN = {'Feature_Name'}; fd = fNames(tIdx); if iscolumn(fd), td = num2cell(fd); else, td = num2cell(fd'); end
                for c = 1:nCols
                    td = [td, num2cell(posImpact(tIdx, c)), num2cell(negImpact(tIdx, c)), num2cell(stdImpact(tIdx, c))]; 
                    if nCols == 1
                        colN = [colN, {'Positive_Marginal_Contrib', 'Negative_Marginal_Contrib', 'Std_Limit'}];
                    else
                        cn = strrep(char(classNamesStr{c}), ' ', '_'); 
                        colN = [colN, {sprintf('%s_Positive_Contrib', cn), sprintf('%s_Negative_Contrib', cn), sprintf('%s_Std', cn)}]; 
                    end
                end
                
                uitable(gS, 'Data', cell2table(td, 'VariableNames', colN));
                uitextarea(gS, 'Value', sumText, 'Editable', 'off', 'FontName', 'Consolas', 'FontSize', 12, 'BackgroundColor', '#f0f8ff');
                close(dlg);
                obj.RightPanel.SelectedTab = tShapTab; obj.addLog(sprintf('[%s] SHAP plotting complete.', mName));
            catch ME, obj.addLog(['Error: ' ME.message]); end
            set(obj.UIFigure, 'Pointer', 'arrow'); drawnow; 
        end
        
        function p = predict_prob(~, mdl, Xn, ci)
            if isa(mdl, 'GeneralizedLinearModel')
                p1 = predict(mdl, Xn);
                sc = [1-p1, p1]; 
                p = sc(:, ci);
            else
                [~, sc] = predict(mdl, Xn); 
                p = sc(:, ci);
            end
        end
        
        function plot_Performance(obj, mName)
            if strcmp(mName, 'Pending Analysis...') || isempty(mName), uialert(obj.UIFigure, 'Select a model.', 'Warning'); return; end
            pd = obj.Results.PlotData;
            m_idx = find(strcmp(pd.models, mName));
            if isempty(m_idx), uialert(obj.UIFigure, 'Data not found.', 'Error'); return; end
            
            t_model = pd.models(m_idx);
            t_L_Int = pd.ML_Labels_Int(m_idx);
            t_P_Int = pd.ML_Probs_Int(m_idx);
            t_Pr_Int = pd.ML_Preds_Int(m_idx);
            t_L_Ext = pd.ML_Labels_Ext(m_idx);
            t_P_Ext = pd.ML_Probs_Ext(m_idx);
            t_Pr_Ext = pd.ML_Preds_Ext(m_idx);
            t_mInt = pd.meanInt(m_idx, :);
            t_sInt = pd.stdInt(m_idx, :);
            t_mExt = pd.meanExt(m_idx, :);
            t_sExt = pd.stdExt(m_idx, :);
            t_iRes = pd.intRes(m_idx, :, :);
            t_eRes = pd.extRes(m_idx, :, :);
            
            obj.Results.PlotFunc(t_model, pd.numC, pd.uClass, t_L_Int, t_P_Int, t_Pr_Int, t_L_Ext, t_P_Ext, t_Pr_Ext, t_mInt, t_sInt, t_mExt, t_sExt, t_iRes, t_eRes);
        end
    end
end