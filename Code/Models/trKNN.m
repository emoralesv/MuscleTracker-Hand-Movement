function [T] = trKNN(feat,idxc,results)

k = 10;
validationAccuracy1 = 0;
validationAccuracy2 = 0;

classificationKNN = fitcknn(...
    feat(:,1:end-1), ...
    feat(:,end), ...
    'Distance', 'Euclidean', ...
    'Exponent', [], ...
    'NumNeighbors', 1, ...
    'DistanceWeight', 'Equal', ...
    'Standardize', true);


partitionedModel = crossval(classificationKNN, 'KFold', k);

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
