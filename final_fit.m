%% 爆炸超压拟合公式 — 最终版
% 三段式 k=6 傅里叶模型 + fminimax 精细优化
%
% 段划分:
%   S1: Z=1,2 (中近场, 共用系数)
%   S2: Z=3   (中场, 独立系数)
%   S3: Z=4   (远场边界, 独立系数 + (1+cosθ)/2 前向增强)
%
% 模型结构: k=6 傅里叶 (cosθ ~ cos6θ), V 线性, 每距离项15参数
% 优化策略: LSQ 粗拟合 → fminimax 极小化最大相对误差
%
% 最终指标:
%   整体 R²=0.9957  MAPE=6.30%  Max=16.38%
%   Z=1: Max=12.24%  Z=2: Max=12.26%
%   Z=3: Max=15.45%  Z=4: Max=16.38%

clear; close all;

%% ====== 1. 数据读取与预处理 ======
opts = detectImportOptions('zhuxing1.xlsx');
opts.VariableNamingRule = 'preserve';
df = readtable('zhuxing1.xlsx', opts);
df.Properties.VariableNames(1:6) = {'M', 'H', 'V', 'P_MPa', 'Z', 'theta'};

T0 = 288.15;
df.T_h = T0 - 6.5 * df.H;
df.Sp = (df.T_h / T0) .^ 5.25588;

%% ====== 2. 静爆15参数基准 ======
df_static = df(df.V == 0, :);
static_model = @(p, X) ...
    (X(:,2).^(2/3)./X(:,1)).*(p(1)+p(2)*cosd(X(:,3))+p(3)*cosd(2*X(:,3))+p(4)*cosd(3*X(:,3))+p(5)*cosd(4*X(:,3)))...
  + (X(:,2).^(1/3)./X(:,1).^2).*(p(6)+p(7)*cosd(X(:,3))+p(8)*cosd(2*X(:,3))+p(9)*cosd(3*X(:,3))+p(10)*cosd(4*X(:,3)))...
  + (1./X(:,1).^3).*(p(11)+p(12)*cosd(X(:,3))+p(13)*cosd(2*X(:,3))+p(14)*cosd(3*X(:,3))+p(15)*cosd(4*X(:,3)));

X_stat = [df_static.Z, df_static.Sp, df_static.theta]; y_stat = df_static.P_MPa;
p0_stat = 0.01*ones(1,15); p0_stat(1)=0.065; p0_stat(6)=0.397; p0_stat(11)=0.322;
[p_stat,~] = lsqcurvefit(static_model, p0_stat, X_stat, y_stat, [], [], ...
    optimoptions('lsqcurvefit','Algorithm','trust-region-reflective','Display','off','MaxFunctionEvaluations',1e5));
p_stat = round(p_stat,4);
disp('=== 静爆15参数 ==='); disp(p_stat);

%% ====== 3. 动爆数据 ======
df_dyn = df(df.V > 0, :);
Z_all = df_dyn.Z; theta_all = df_dyn.theta; V_all = df_dyn.V; H_all = df_dyn.H;
X_all = [df_dyn.Z, df_dyn.Sp, df_dyn.theta, df_dyn.V];
y_all = df_dyn.P_MPa;

%% ====== 4. 33参模型 (初始化用) ======
dyn33 = @(c,X) (X(:,2).^(2/3)./X(:,1)).*(c(1)+c(2)*(X(:,4)/1000)+c(3)*(X(:,4)/1000).^2+(c(4)+c(5)*(X(:,4)/1000)).*cosd(X(:,3))+(c(6)+c(7)*(X(:,4)/1000)).*cosd(2*X(:,3))+(c(8)+c(9)*(X(:,4)/1000)).*cosd(3*X(:,3))+(c(10)+c(11)*(X(:,4)/1000)).*cosd(4*X(:,3)))+(X(:,2).^(1/3)./X(:,1).^2).*(c(12)+c(13)*(X(:,4)/1000)+c(14)*(X(:,4)/1000).^2+(c(15)+c(16)*(X(:,4)/1000)).*cosd(X(:,3))+(c(17)+c(18)*(X(:,4)/1000)).*cosd(2*X(:,3))+(c(19)+c(20)*(X(:,4)/1000)).*cosd(3*X(:,3))+(c(21)+c(22)*(X(:,4)/1000)).*cosd(4*X(:,3)))+(1./X(:,1).^3).*(c(23)+c(24)*(X(:,4)/1000)+c(25)*(X(:,4)/1000).^2+(c(26)+c(27)*(X(:,4)/1000)).*cosd(X(:,3))+(c(28)+c(29)*(X(:,4)/1000)).*cosd(2*X(:,3))+(c(30)+c(31)*(X(:,4)/1000)).*cosd(3*X(:,3))+(c(32)+c(33)*(X(:,4)/1000)).*cosd(4*X(:,3)));
c33 = lsqcurvefit(dyn33, [p_stat(1),0.05*ones(1,10),p_stat(6),0.05*ones(1,10),p_stat(11),0.05*ones(1,10)], X_all, y_all, [], [], optimoptions('lsqcurvefit','Algorithm','trust-region-reflective','Display','off','MaxFunctionEvaluations',5e5));
c33 = round(c33,4);

%% ====== 5. k=6 45参数模型 ======
dyn45_k6 = @(c,X) ...
    (X(:,2).^(2/3)./X(:,1)).*(c(1)+c(2)*(X(:,4)/1000)+c(3)*(X(:,4)/1000).^2+...
        (c(4)+c(5)*(X(:,4)/1000)).*cosd(X(:,3))+(c(6)+c(7)*(X(:,4)/1000)).*cosd(2*X(:,3))+...
        (c(8)+c(9)*(X(:,4)/1000)).*cosd(3*X(:,3))+(c(10)+c(11)*(X(:,4)/1000)).*cosd(4*X(:,3))+...
        (c(12)+c(13)*(X(:,4)/1000)).*cosd(5*X(:,3))+(c(14)+c(15)*(X(:,4)/1000)).*cosd(6*X(:,3)))...
  + (X(:,2).^(1/3)./X(:,1).^2).*(c(16)+c(17)*(X(:,4)/1000)+c(18)*(X(:,4)/1000).^2+...
        (c(19)+c(20)*(X(:,4)/1000)).*cosd(X(:,3))+(c(21)+c(22)*(X(:,4)/1000)).*cosd(2*X(:,3))+...
        (c(23)+c(24)*(X(:,4)/1000)).*cosd(3*X(:,3))+(c(25)+c(26)*(X(:,4)/1000)).*cosd(4*X(:,3))+...
        (c(27)+c(28)*(X(:,4)/1000)).*cosd(5*X(:,3))+(c(29)+c(30)*(X(:,4)/1000)).*cosd(6*X(:,3)))...
  + (1./X(:,1).^3).*(c(31)+c(32)*(X(:,4)/1000)+c(33)*(X(:,4)/1000).^2+...
        (c(34)+c(35)*(X(:,4)/1000)).*cosd(X(:,3))+(c(36)+c(37)*(X(:,4)/1000)).*cosd(2*X(:,3))+...
        (c(38)+c(39)*(X(:,4)/1000)).*cosd(3*X(:,3))+(c(40)+c(41)*(X(:,4)/1000)).*cosd(4*X(:,3))+...
        (c(42)+c(43)*(X(:,4)/1000)).*cosd(5*X(:,3))+(c(44)+c(45)*(X(:,4)/1000)).*cosd(6*X(:,3)));

%% ====== 6. 33→45初始化 ======
c0_45 = zeros(1,45);
for t = 0:2
    c0_45(t*15+(1:3)) = c33(t*11+(1:3));
    c0_45(t*15+(4:11)) = c33(t*11+(4:11));
end

%% ====== 7. 段定义 + 拟合 ======
mask_S1 = (Z_all == 1 | Z_all == 2);
mask_S2 = (Z_all == 3);
mask_S3 = (Z_all == 4);

opts_lsq = optimoptions('lsqcurvefit','Algorithm','trust-region-reflective','Display','off',...
    'MaxFunctionEvaluations',1e6,'MaxIterations',10000,'FunctionTolerance',1e-12);
opts_mm = optimoptions('fminimax','MaxFunctionEvaluations',2e6,'MaxIterations',10000,...
    'OptimalityTolerance',1e-8,'StepTolerance',1e-8,'Display','off');

% --- S1: Z=1,2 ---
fprintf('拟合 S1 (Z=1,2)...\n');
[cS1,~] = lsqcurvefit(dyn45_k6, c0_45, X_all(mask_S1,:), y_all(mask_S1), [], [], opts_lsq);
cS1 = round(cS1,4);
rel_S1 = @(c) abs(dyn45_k6(c, X_all(mask_S1,:)) - y_all(mask_S1)) ./ y_all(mask_S1) * 100;
[cS1,~] = fminimax(rel_S1, cS1, [],[],[],[],[],[],[], opts_mm);
cS1 = round(cS1,4);
errS1 = rel_S1(cS1);
fprintf('  S1 fminimax: Max=%.2f%% MAPE=%.2f%%\n', max(errS1), mean(errS1));

% --- S2: Z=3 ---
fprintf('拟合 S2 (Z=3)...\n');
[cS2,~] = lsqcurvefit(dyn45_k6, c0_45, X_all(mask_S2,:), y_all(mask_S2), [], [], opts_lsq);
cS2 = round(cS2,4);
rel_S2 = @(c) abs(dyn45_k6(c, X_all(mask_S2,:)) - y_all(mask_S2)) ./ y_all(mask_S2) * 100;
[cS2,~] = fminimax(rel_S2, cS2, [],[],[],[],[],[],[], opts_mm);
cS2 = round(cS2,4);
errS2 = rel_S2(cS2);
fprintf('  S2 fminimax: Max=%.2f%% MAPE=%.2f%%\n', max(errS2), mean(errS2));

% --- S3: Z=4 (k=6 + 前向boost) ---
fprintf('拟合 S3 (Z=4, boost)...\n');
Xs = X_all(mask_S3,:); ys = y_all(mask_S3);
cS3_k6 = lsqcurvefit(dyn45_k6, c0_45, Xs, ys, [], [], opts_lsq);
cS3_k6 = round(cS3_k6,4);

dyn_boosted = @(c,X) ...
    dyn45_k6(c(1:45), X) ...
    + (X(:,2).^(2/3)./X(:,1)).*((X(:,4)/1000).*(c(46)+c(47)*(1+cosd(X(:,3)))/2+c(48)*((1+cosd(X(:,3)))/2).^2) ...
                               + (X(:,4)/1000).^2.*(c(49)+c(50)*(1+cosd(X(:,3)))/2)) ...
    + (1./X(:,1).^3).*((X(:,4)/1000).*(c(51)+c(52)*(1+cosd(X(:,3)))/2+c(53)*((1+cosd(X(:,3)))/2).^2) ...
                      + (X(:,4)/1000).^2.*(c(54)+c(55)*(1+cosd(X(:,3)))/2));

c0_boost = [cS3_k6, zeros(1,10)];
opts_boost = optimoptions('lsqcurvefit','Algorithm','trust-region-reflective','Display','off',...
    'MaxFunctionEvaluations',2e6,'MaxIterations',20000,'FunctionTolerance',1e-14);
[cS3_lsq,~] = lsqcurvefit(dyn_boosted, c0_boost, Xs, ys, [], [], opts_boost);
cS3_lsq = round(cS3_lsq,4);
rel_S3 = @(c) abs(dyn_boosted(c, Xs) - ys) ./ ys * 100;
[cS3,~] = fminimax(rel_S3, cS3_lsq, [],[],[],[],[],[],[], opts_mm);
cS3 = round(cS3,4);
errS3 = rel_S3(cS3);
fprintf('  S3 fminimax: Max=%.2f%% MAPE=%.2f%%\n', max(errS3), mean(errS3));

%% ====== 8. 合并评估 ======
P_final = zeros(size(y_all));
P_final(mask_S1) = dyn45_k6(cS1, X_all(mask_S1,:));
P_final(mask_S2) = dyn45_k6(cS2, X_all(mask_S2,:));
P_final(mask_S3) = dyn_boosted(cS3, X_all(mask_S3,:));
err_final = abs(P_final - y_all) ./ y_all * 100;
R2 = 1 - sum((y_all-P_final).^2)/sum((y_all-mean(y_all)).^2);

fprintf('\n========== 最终结果 ==========\n');
fprintf('整体: R²=%.4f  MAPE=%.2f%%  Max=%.2f%%\n', R2, mean(err_final), max(err_final));
for z = 1:4
    m = (Z_all == z); e = err_final(m);
    fprintf('  Z=%d: MAPE=%.2f%%  Max=%.2f%%  P90=%.2f%%\n', z, mean(e), max(e), prctile(e,90));
end

%% ====== 9. 保存系数 ======
save('final_coefficients.mat', 'p_stat', 'cS1', 'cS2', 'cS3');
disp('=== 系数已保存至 final_coefficients.mat ===');

%% ====== 10. 输出公式文本 ======
fprintf('\n========== 最终公式 ==========\n');
fprintf('P_dyn(Z,V,theta,Sp) = T1 + T2 + T3  [+ Boost for Z=4]\n');
fprintf('Vn = V/1000,  Sp = (T_h/T0)^5.25588,  T_h = T0 - 6.5*H\n\n');

% 通用打印函数
print_term = @(c, start_idx, name) fprintf('%s: const=%.4f%+.4fVn%+.4fVn² | cosθ=%.4f%+.4fVn | cos2θ=%.4f%+.4fVn | cos3θ=%.4f%+.4fVn | cos4θ=%.4f%+.4fVn | cos5θ=%.4f%+.4fVn | cos6θ=%.4f%+.4fVn\n', ...
    name, c(start_idx), c(start_idx+1), c(start_idx+2), c(start_idx+3), c(start_idx+4), ...
    c(start_idx+5), c(start_idx+6), c(start_idx+7), c(start_idx+8), c(start_idx+9), c(start_idx+10), ...
    c(start_idx+11), c(start_idx+12), c(start_idx+13), c(start_idx+14));

fprintf('--- 段S1 (Z=1,2) ---\n');
print_term(cS1, 1, 'T1=(Sp^(2/3)/Z)×');
print_term(cS1, 16, 'T2=(Sp^(1/3)/Z²)×');
print_term(cS1, 31, 'T3=(1/Z³)×');

fprintf('\n--- 段S2 (Z=3) ---\n');
print_term(cS2, 1, 'T1=(Sp^(2/3)/Z)×');
print_term(cS2, 16, 'T2=(Sp^(1/3)/Z²)×');
print_term(cS2, 31, 'T3=(1/Z³)×');

fprintf('\n--- 段S3 (Z=4) ---\n');
fprintf('Base (k=6):\n');
print_term(cS3, 1, 'T1=(Sp^(2/3)/Z)×');
print_term(cS3, 16, 'T2=(Sp^(1/3)/Z²)×');
print_term(cS3, 31, 'T3=(1/Z³)×');
fprintf('Boost (前向增强):\n');
fprintf('  B1=(Sp^(2/3)/Z)×[%.4fVn + %.4fVn·f(θ) + %.4fVn·f²(θ) + %.4fVn² + %.4fVn²·f(θ)]\n', cS3(46:50));
fprintf('  B2=(1/Z³)×     [%.4fVn + %.4fVn·f(θ) + %.4fVn·f²(θ) + %.4fVn² + %.4fVn²·f(θ)]\n', cS3(51:55));
fprintf('  其中 f(θ) = (1+cosθ)/2\n');

%% ====== 11. 生成 LaTeX 代码 ======
fprintf('\n========== LaTeX 代码 ==========\n');
generate_latex(p_stat, cS1, cS2, cS3);

%% ====== 12. 导出 Excel ======
df_out = df_dyn(:, {'M','H','V','Z','theta','P_MPa'});
df_out.P_pred = P_final;
df_out.Abs_Error = abs(P_final - y_all);
df_out.Rel_Error_Pct = err_final;
df_out.Segment = cell(height(df_dyn),1);
df_out.Segment(mask_S1) = {'S1_Z12'};
df_out.Segment(mask_S2) = {'S2_Z3'};
df_out.Segment(mask_S3) = {'S3_Z4'};
df_out.Properties.VariableNames{'P_pred'} = 'P_拟合计算值_MPa';
df_out.Properties.VariableNames{'Abs_Error'} = '绝对误差_MPa';
df_out.Properties.VariableNames{'Rel_Error_Pct'} = '相对误差_百分比';
df_out.Properties.VariableNames{'Segment'} = '所属分段';
writetable(sortrows(df_out, '相对误差_百分比', 'descend'), '最终结果_按误差排序.xlsx');
writetable(sortrows(df_out, {'H','V','Z','theta'}), '最终结果_按工况排序.xlsx');
disp('=== Excel 已导出 ===');

%% ====== LaTeX 生成函数 ======
function generate_latex(p_stat, cS1, cS2, cS3)
    fid = fopen('formula_latex.tex', 'w');

    fprintf(fid, '%% 爆炸超压拟合公式 — LaTeX 代码\n');
    fprintf(fid, '%% 自动生成, 可直接插入 Word (用 LaTeX 模式) 或复制到 LaTeX 编辑器\n\n');

    % 环境定义
    fprintf(fid, '\\section*{爆炸冲击波超压峰值经验公式}\n\n');

    fprintf(fid, '\\subsection*{变量定义}\n');
    fprintf(fid, '\\begin{align*}\n');
    fprintf(fid, '  V_n &= \\frac{V}{1000} \\quad (\\text{归一化速度, m/s}) \\\\\n');
    fprintf(fid, '  T_h &= T_0 - 6.5H \\quad (\\text{海拔修正温度, K}) \\\\\n');
    fprintf(fid, '  S_p &= \\left(\\frac{T_h}{T_0}\\right)^{5.25588} \\quad (\\text{萨克斯缩放因子}) \\\\\n');
    fprintf(fid, '  T_0 &= 288.15\\,\\mathrm{K} \\quad (\\text{海平面标准温度})\n');
    fprintf(fid, '\\end{align*}\n\n');

    % 通用公式形式
    fprintf(fid, '\\subsection*{通用公式结构}\n');
    fprintf(fid, '\\begin{equation}\n');
    fprintf(fid, '  P_{\\mathrm{dyn}}(Z, S_p, \\theta, V) = T_1 + T_2 + T_3 + B \\cdot \\mathbf{1}_{Z=4}\n');
    fprintf(fid, '\\end{equation}\n');
    fprintf(fid, '\\begin{align*}\n');
    fprintf(fid, '  T_1 &= \\frac{S_p^{2/3}}{Z} \\sum_{k=0}^{6} (a_{1,k} + b_{1,k} V_n) \\cos(k\\theta) \\\\\n');
    fprintf(fid, '  T_2 &= \\frac{S_p^{1/3}}{Z^2} \\sum_{k=0}^{6} (a_{2,k} + b_{2,k} V_n) \\cos(k\\theta) \\\\\n');
    fprintf(fid, '  T_3 &= \\frac{1}{Z^3} \\sum_{k=0}^{6} (a_{3,k} + b_{3,k} V_n) \\cos(k\\theta)\n');
    fprintf(fid, '\\end{align*}\n\n');

    fprintf(fid, '\\textbf{使用规则:}\n');
    fprintf(fid, '\\begin{itemize}\n');
    fprintf(fid, '  \\item $Z \\leq 2.5$: 使用段S1系数 (中近场, Z=1,2)\n');
    fprintf(fid, '  \\item $2.5 < Z \\leq 3.5$: 使用段S2系数 (中场, Z=3)\n');
    fprintf(fid, '  \\item $Z > 3.5$: 使用段S3系数 (远场边界, Z=4, 含前向增强项B)\n');
    fprintf(fid, '\\end{itemize}\n\n');

    % 打印系数表
    print_coeff_table_latex(fid, 'S1 (Z=1,2)', cS1, false);
    print_coeff_table_latex(fid, 'S2 (Z=3)', cS2, false);
    print_coeff_table_latex(fid, 'S3 (Z=4, 含Boost)', cS3, true);

    % 静爆公式
    fprintf(fid, '\\subsection*{静爆基准公式 (V=0)}\n');
    fprintf(fid, '\\begin{equation}\n');
    fprintf(fid, '  P_{\\mathrm{stat}}(Z, S_p, \\theta) = ');
    fprintf(fid, '\\frac{S_p^{2/3}}{Z}\\left[%.4f%+.4f\\cos\\theta%+.4f\\cos2\\theta%+.4f\\cos3\\theta%+.4f\\cos4\\theta\\right]\n', p_stat(1:5));
    fprintf(fid, '  + \\frac{S_p^{1/3}}{Z^2}\\left[%.4f%+.4f\\cos\\theta%+.4f\\cos2\\theta%+.4f\\cos3\\theta%+.4f\\cos4\\theta\\right]\n', p_stat(6:10));
    fprintf(fid, '  + \\frac{1}{Z^3}\\left[%.4f%+.4f\\cos\\theta%+.4f\\cos2\\theta%+.4f\\cos3\\theta%+.4f\\cos4\\theta\\right]\n', p_stat(11:15));
    fprintf(fid, '\\end{equation}\n\n');

    fclose(fid);
    fprintf('LaTeX 代码已保存至 formula_latex.tex\n');
end

function print_coeff_table_latex(fid, seg_name, c, has_boost)
    fprintf(fid, '\\subsubsection*{段%s 系数}\n', seg_name);
    fprintf(fid, '\\begin{table}[h]\n');
    fprintf(fid, '  \\centering\n');
    fprintf(fid, '  \\caption{段%s 拟合系数 ($k=0$ 为常数项)}\n', seg_name);
    fprintf(fid, '  \\begin{tabular}{c|rrr|rrr|rrr}\n');
    fprintf(fid, '    \\hline\n');
    fprintf(fid, '    \\multicolumn{1}{c|}{} & \\multicolumn{3}{c|}{$T_1$ ($S_p^{2/3}/Z$)} & ');
    fprintf(fid, '\\multicolumn{3}{c|}{$T_2$ ($S_p^{1/3}/Z^2$)} & \\multicolumn{3}{c}{$T_3$ ($1/Z^3$)} \\\\\n');
    fprintf(fid, '    $k$ & $a_{1,k}$ & $b_{1,k}$ & $c_{1,k}$ & $a_{2,k}$ & $b_{2,k}$ & $c_{2,k}$ & $a_{3,k}$ & $b_{3,k}$ & $c_{3,k}$ \\\\\n');
    fprintf(fid, '    \\hline\n');

    cos_labels = {'0 (const)', '\\cos\\theta', '\\cos2\\theta', '\\cos3\\theta', '\\cos4\\theta', '\\cos5\\theta', '\\cos6\\theta'};

    for k = 0:6
        if k == 0
            % 常数项: 有 Vn² 项
            fprintf(fid, '    $%s$ & %.4f & %.4f & %.4f & %.4f & %.4f & %.4f & %.4f & %.4f & %.4f \\\\\n', ...
                cos_labels{k+1}, ...
                c(1), c(2), c(3), c(16), c(17), c(18), c(31), c(32), c(33));
        else
            % 角度项: 无 Vn²
            base1 = 1 + 3 + (k-1)*2;
            base2 = 16 + 3 + (k-1)*2;
            base3 = 31 + 3 + (k-1)*2;
            fprintf(fid, '    $%s$ & %.4f & %.4f & -- & %.4f & %.4f & -- & %.4f & %.4f & -- \\\\\n', ...
                cos_labels{k+1}, ...
                c(base1), c(base1+1), c(base2), c(base2+1), c(base3), c(base3+1));
        end
    end
    fprintf(fid, '    \\hline\n');
    fprintf(fid, '  \\end{tabular}\n');
    fprintf(fid, '\\end{table}\n\n');

    % Boost 系数 (仅 Z=4)
    if has_boost
        fprintf(fid, '\\textbf{前向增强项 (仅Z=4):}\n');
        fprintf(fid, '\\begin{equation}\n');
        fprintf(fid, '  B = \\frac{S_p^{2/3}}{Z}\\left[V_n\\left(%.4f%+.4f f(\\theta)%+.4f f^2(\\theta)\\right) + V_n^2\\left(%.4f%+.4f f(\\theta)\\right)\\right]\n', c(46:50));
        fprintf(fid, '  + \\frac{1}{Z^3}\\left[V_n\\left(%.4f%+.4f f(\\theta)%+.4f f^2(\\theta)\\right) + V_n^2\\left(%.4f%+.4f f(\\theta)\\right)\\right]\n', c(51:55));
        fprintf(fid, '\\end{equation}\n');
        fprintf(fid, '其中 $f(\\theta) = \\frac{1+\\cos\\theta}{2}$ 为前向定向函数。\n\n');
    end
end
