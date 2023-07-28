%%Proyecto 2do Parcial 
% John Bullinaria's Implementing a Neural Network in matlab
% Moctezuma Peralta Karen Yudit
% Torres Montes Mariana
% ISC F64

clear
clc 

PatronEntre = 4;
NumEntra = 2;
Paramet = 2;
UniSali = 1;

MatrizEntra = [0, 0;0, 1;1, 0;1, 1];
Obje = [0;1;1;0];

          
          
SumH = zeros(PatronEntre+1, Paramet+1);
WeightIH = 2 * (rand(NumEntra+1, Paramet+1) - 0.5);
Hidden = zeros(PatronEntre+1, Paramet+1);

SumO = zeros(PatronEntre+1, UniSali+1);
WeightHO = 2 * (rand(Paramet+1, UniSali+1) - 0.5);
Output = zeros(PatronEntre+1, UniSali+1);

DeltaO = zeros(UniSali+1, 1);
SumDOW = zeros(Paramet+1, 1);
DeltaH = zeros(Paramet+1, 1);

eta = 0.5;
alpha = 0.9;
smallwt = 0.5;

for j = 1:Paramet
    for i = 1:NumEntra+1
        DeltaWeightIH(i, j) = 0.0;
        WeightIH(i, j) = 2.0 * (rand - 0.5) * smallwt;
    end
end

for k = 1:UniSali
    for j = 1:Paramet+1
        DeltaWeightHO(j, k) = 0.0;
        WeightHO(j, k) = 2.0 * (rand - 0.5) * smallwt;
    end
end

for epoch = 0:99999
    ranpat = randperm(PatronEntre);
    Error = 0.0;

    for np = 1:PatronEntre
        p = ranpat(np);
        for j = 1:Paramet
            SumH(p, j) = WeightIH(1, j);
            for i = 1:NumEntra
                SumH(p, j) = SumH(p, j) + MatrizEntra(p, i) * WeightIH(i, j);
            end
            Hidden(p, j) = 1.0 / (1.0 + exp(-SumH(p, j)));
        end
        for k = 1:UniSali
            SumO(p, k) = WeightHO(1, k);
            for j = 1:Paramet
                SumO(p, k) = SumO(p, k) + Hidden(p, j) * WeightHO(j, k);
            end
            Output(p, k) = 1.0 / (1.0 + exp(-SumO(p, k)));
            Error = Error + 0.5 * (Obje(p, k) - Output(p, k))^2;
            DeltaO(k) = (Obje(p, k) - Output(p, k)) * Output(p, k) * (1.0 - Output(p, k));
        end
        for j = 1:Paramet
            SumDOW(j) = 0.0;
            for k = 1:UniSali
                SumDOW(j) = SumDOW(j) + WeightHO(j, k) * DeltaO(k);
            end
            DeltaH(j) = SumDOW(j) * Hidden(p, j) * (1.0 - Hidden(p, j));
        end
        for j = 1:Paramet
            DeltaWeightIH(1, j) = eta * DeltaH(j) + alpha * DeltaWeightIH(1, j);
            WeightIH(1, j) = WeightIH(1, j) + DeltaWeightIH(1, j);
            for i = 2:NumEntra+1
                DeltaWeightIH(i, j) = eta * MatrizEntra(p, i-1) * DeltaH(j) + alpha * DeltaWeightIH(i, j);
                WeightIH(i, j) = WeightIH(i, j) + DeltaWeightIH(i, j);
            end
        end
        for k = 1:UniSali
            DeltaWeightHO(1, k) = eta * DeltaO(k) + alpha * DeltaWeightHO(1, k);
            WeightHO(1, k) = WeightHO(1, k) + DeltaWeightHO(1, k);
            for j = 2:Paramet+1
                DeltaWeightHO(j, k) = eta * Hidden(p, j-1) * DeltaO(k) + alpha * DeltaWeightHO(j, k);
                WeightHO(j, k) = WeightHO(j, k) + DeltaWeightHO(j, k);
            end
        end
    end

    if mod(epoch, 100) == 0
        fprintf('\nEpoch %-5d :   Error = %f', epoch, Error);
    end

    if Error < 0.0004
        break;
    end
end

fprintf('\n\nDatos de las epocas: %d\n\nPat\t', epoch);
for i = 1:NumEntra
    fprintf('Input%-4d\t', i);
end
for k = 1:UniSali
    fprintf('Target%-4d\tOutput%-4d\t', k, k);
end
for p = 1:PatronEntre
    fprintf('\n%d\t', p);
    for i = 1:NumEntra
        fprintf('%f\t', MatrizEntra(p, i));
    end
    for k = 1:UniSali
        fprintf('%f\t%f\t', Obje(p, k), Output(p, k));
    end
end

fprintf('\n\n Proyecto 2do Parcial \n');
fprintf('\n Moctezuma Peralta Karen Yudit \n');
fprintf('\n Torres Montes Mariana \n');
fprintf('\n Goodbye Teacher \n');
clear