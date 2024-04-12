function [T] = trNN(feat,idxc,results)

k = 10;
validationAccuracy1 = 0;
validationAccuracy2 = 0;

classificationNeuralNetwork = fitcnet(...
      feat(:,1:end-1), ...
      feat(:,end), ...
    'LayerSizes', 100, ...
    'Activations', 'relu', ...
    'Lambda', 0, ...
    'IterationLimit', 1000, ...
    'Standardize', true);


partitionedModel = crossval(classificationNeuralNetwork, 'KFold', k);
% Compute validation predictions
[validationPredictions, validationScores] = kfoldPredict(partitionedModel);

for i = 1: k
    idx =  partitionedModel.Partition.test(i);
    cc = unique(idxc);
    for c = 1: size(cc,1)
        cp = classperf(feat(idx==(idxc==cc(c)),end),validationPredictions(idx==(idxc==cc(c))));
        if c == 1
            validationAccuracy1 = cp.CorrectRate;
        else
            validationAccuracy2 = cp.CorrectRate;
        end

    end
    cp = classperf(feat(idx,end),validationPredictions(idx));
    bothAccuracy = cp.CorrectRate;
    T(i,:) = {results{1},results{2},results{3},results{4},validationAccuracy1,validationAccuracy2,bothAccuracy};
end
end
