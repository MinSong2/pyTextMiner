import pyTextMiner as ptm

test='이건진짜좋은영화라라랜드진짜좋은영화'

model_path='./model/korean_segmentation_model.crfsuite'
segmentation=ptm.segmentation.SegmentationKorean(model_path)
correct=segmentation(test)
print(correct)

chatspace_segmentation=ptm.segmentation.ChatSpaceSegmentationKorean()
chatspace_correct=chatspace_segmentation(test)
print(chatspace_correct)

lstm_model_path='./pyTextMiner/segmentation/model'
lstm_segmentation=ptm.segmentation.LSTMSegmentationKorean(lstm_model_path)
lstm_correct=lstm_segmentation(test)
print(lstm_correct)

lstm_segmentation.close()