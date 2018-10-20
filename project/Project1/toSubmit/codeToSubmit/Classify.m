function OutLabels = Classify(TrainSet,TestSet,LabelsTrain)
%[OutLabelsTrain,OutLabelsTest] = Classify(TrainSet,TestSet,LabelsTrain)
%   Classify high dimensional data using an SVM and return the labels
% for tests = 1:length(TrainSet)
%     Mdl = fitclinear(TrainSet{tests},LabelsTrain{tests});
%     OutLabelsTrain{tests} = predict(Mdl,TrainSet{tests});
%     OutLabelsTest{tests} = predict(Mdl,TestSet{tests});
% end
Mdl = fitcsvm(TrainSet, LabelsTrain);
OutLabelsTrain = predict(Mdl, TrainSet);
OutLabelsTest  = predict(Mdl, TestSet);
OutLabels.Train = OutLabelsTrain;
OutLabels.Test = OutLabelsTest;
end
