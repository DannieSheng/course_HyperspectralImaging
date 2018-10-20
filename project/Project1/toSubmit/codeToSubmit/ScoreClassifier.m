function PercCorrect = ScoreClassifier(TrueLabels,OutLabels)
%Get percent correct of the labels from the classifier compared with the
%true labels
% (NX1) -1 or 1

OutLabelsTrain  = OutLabels.Train;
OutLabelsTrain(OutLabelsTrain == -1) = 0;
OutLabelsTest   = OutLabels.Test;
OutLabelsTest(OutLabelsTest == -1) = 0;
TrueLabelsTrain = TrueLabels.Train;
TrueLabelsTrain(TrueLabelsTrain == -1) = 0;
TrueLabelsTest  = TrueLabels.Test;
TrueLabelsTest(TrueLabelsTest == -1) = 0;

trainCorrect = 1-sum(abs(OutLabelsTrain-TrueLabelsTrain))/length(TrueLabelsTrain);
testCorrect  = 1-sum(abs(OutLabelsTest-TrueLabelsTest))/length(TrueLabelsTest);

PercCorrect.train = trainCorrect;
PercCorrect.test = testCorrect;
end

