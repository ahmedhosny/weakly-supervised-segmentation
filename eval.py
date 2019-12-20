import os
import torch
import scipy.io as scio
import numpy as np

class Metric():
    def __init__(self, reduction='mean'):
        self.reduciton = reduction

    def set_input(self, y_true, y_pred):
        '''

        :param y_true: torch.Tensor [channel, h, w] or [batch, channel, h, w] in range of [0,num_class-1]
        :param y_pred: torch.Tensor [channel, h, w] or [batch,channel, h, w] in range of [0,num_class-1]
        :return:
        '''
        self.y_true_batch = None  # for the batch-format y_true
        self.y_pred_batch = None  # for the batch-format y_pred
        self.y_true_case = None  # for the case-format y_true
        self.y_pred_case = None  # for the case-format y_pred
        self.shape = y_true.shape
        self.num_class = 3

        if len(self.shape) == 4:  # the batch-format
            assert y_true.shape == y_pred.shape, 'the shape of y_true and y_pred should be same!'
            self.y_true_batch = y_true
            self.y_pred_batch = y_pred

        if len(self.shape) == 3:  # the no batch-fromat
            assert y_true.shape == y_pred.shape, 'the shape of y_true and y_pred should be same!'
            self.y_true_case = y_true
            self.y_pred_case = y_pred

    def dice_for_case(self):

        # out = torch.rand(self.num_class)
        out = np.zeros((self.num_class))
        assert len(self.y_pred_case.shape) == 3, 'the input for dice_for_batch should has 3 dims'

        try:
            # Compute tumor+kidney Dice
            tk_pd = np.greater(self.y_pred_case, 0)
            tk_gt = np.greater(self.y_true_case, 0)
            tk_dice = 2 * np.logical_and(tk_pd, tk_gt).sum() / (
                    tk_pd.sum() + tk_gt.sum() + 1e-5
            )
        except ZeroDivisionError:
            return 0.0, 0.0

        try:
            # Compute tumor Dice
            tu_pd = np.greater(self.y_pred_case, 1)
            tu_gt = np.greater(self.y_true_case, 1)
            tu_dice = 2 * np.logical_and(tu_pd, tu_gt).sum() / (
                    tu_pd.sum() + tu_gt.sum() + 1e-5
            )
        except ZeroDivisionError:
            return tk_dice, 0.0
        out[0] = tk_dice
        out[1] = tu_dice
        return out

    def dice_for_batch(self):
        assert len(self.shape) == 4, 'the input for dice_for_batch should has 4 dims'
        out = np.zeros((self.shape[0], self.num_class))

        for batch_index in range(self.shape[0]):
            self.y_true_case = self.y_true_batch[batch_index]
            self.y_pred_case = self.y_pred_batch[batch_index]
            out[batch_index] = self.dice_for_case()

        return out


if __name__ == '__main__':
    # Test different results generated by models
    # Change the path to test different results
    cams_name = 'saved/kidney_cls/cams/'
    gt_name = 'saved/kidney_cls/gt/'
    file_names = os.listdir(cams_name)
    predictions = []
    labels = []
    for idx, file_name in enumerate(file_names):
        cam = np.load(os.path.join(cams_name, file_name))
        if len(cam.item()) == 0:
            continue

        norm_cam = [cam for key, cam in cam.item().items()]
        norm_cam = np.array(norm_cam)
        if norm_cam.shape[0] == 256:
            norm_cam = norm_cam[np.newaxis, :, :]
        bg_score = [np.ones_like(norm_cam[0]) * 0.2]
        pred = np.argmax(np.concatenate((bg_score, norm_cam)), 0)

        gt = np.load(os.path.join(gt_name, file_name))
        predictions.append(pred)
        labels.append(gt)
        if (idx+1) % 100 == 0:
            print('[{}/{}] completed.'.format(idx+1, len(file_names)))

    predictions, labels = np.array(predictions), np.array(labels)
    print(predictions.shape)
    metricer = Metric()
    metricer.set_input(labels, predictions)
    print('test for the dice_for_case')
    print('dice: ', metricer.dice_for_case())

    # Test the results of post-processing
    dataFile = 'result_final.mat'
    gt_name = 'saved/kidney_cls/gt/'
    data = scio.loadmat(dataFile)
    predictions = []
    labels = []
    for i in range(7898):
        index = data['index'][i][0][0][:-4]
        cam = data['result'][i][0]
        predictions.append(cam)

        gt = np.load(os.path.join(gt_name, index + '.npy'))
        labels.append(gt)
        if (i + 1) % 100 == 0:
            print('[{}/{}] completed.'.format(i + 1, 7898))

    predictions, labels = np.array(predictions), np.array(labels)
    print(predictions.shape)
    metricer = Metric()
    metricer.set_input(labels, predictions)
    print('test for the dice_for_case')
    print('dice: ', metricer.dice_for_case())