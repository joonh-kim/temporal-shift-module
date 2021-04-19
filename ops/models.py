# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

from torch import nn

from ops.basic_ops import ConsensusModule
from ops.transforms import *
from torch.nn.init import normal_, constant_
from ops.utils import DCTmatrix, DCTmatrix_hat

def init_hidden(batch_size, cell_size):
    init_cell = torch.Tensor(batch_size, cell_size).zero_()
    if torch.cuda.is_available():
        init_cell = init_cell.cuda()
    return init_cell

class TSN(nn.Module):
    def __init__(self, num_class, num_segments, modality,
                 base_model='resnet101', new_length=1,
                 consensus_type='avg', before_softmax=True,
                 dropout=0.5, img_feature_dim=256,
                 crop_num=1, partial_bn=True, print_spec=True, pretrain='imagenet',
                 fc_lr5=False, two_stream=False):
        super(TSN, self).__init__()
        self.num_class = num_class
        self.modality = modality
        self.num_segments = num_segments
        self.before_softmax = before_softmax
        self.dropout = dropout
        self.crop_num = crop_num
        self.consensus_type = consensus_type
        self.img_feature_dim = img_feature_dim  # the dimension of the CNN feature to represent each frame
        self.pretrain = pretrain

        self.base_model_name = base_model
        self.fc_lr5 = fc_lr5

        self.two_stream = two_stream

        if not before_softmax and consensus_type != 'avg':
            raise ValueError("Only avg consensus can be used after Softmax")

        self.new_length = new_length

        if print_spec:
            print(("""
            Initializing TSN with base model: {}.
            TSN Configurations:
                num_segments:       {}                
                dropout_ratio:      {}
                img_feature_dim:    {}
                two_stream:         {}
            """.format(base_model, self.num_segments, self.dropout,
                       self.img_feature_dim, self.two_stream)))

        self._prepare_base_model(base_model)
        self._prepare_tsn(num_class)
        self.consensus = ConsensusModule(consensus_type)

        if self.two_stream:
            self._prepare_freq_model(base_model)
            self._prepare_freq_tsn(num_class)

            self.DCT = DCTmatrix(num_segments)
            self.DCT_hat = DCTmatrix_hat(self.DCT, num_segments)
            self.DCT, self.DCT_hat = \
                torch.from_numpy(self.DCT).type(torch.float32), torch.from_numpy(self.DCT_hat).type(torch.float32)

        if not self.before_softmax:
            self.softmax = nn.Softmax()

        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)

    def _prepare_tsn(self, num_class):
        feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features
        if self.dropout == 0:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Linear(feature_dim, num_class))
            self.new_fc = None
        else:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=self.dropout))
            self.new_fc = nn.Linear(feature_dim, num_class)

        std = 0.001
        if self.new_fc is None:
            normal_(getattr(self.base_model, self.base_model.last_layer_name).weight, 0, std)
            constant_(getattr(self.base_model, self.base_model.last_layer_name).bias, 0)
        else:
            if hasattr(self.new_fc, 'weight'):
                normal_(self.new_fc.weight, 0, std)
                constant_(self.new_fc.bias, 0)

    def _prepare_freq_tsn(self, num_class):
        feature_dim = getattr(self.freq_model, self.freq_model.last_layer_name).in_features
        if self.dropout == 0:
            setattr(self.freq_model, self.freq_model.last_layer_name, nn.Linear(feature_dim, num_class))
            self.freq_new_fc = None
        else:
            setattr(self.freq_model, self.freq_model.last_layer_name, nn.Dropout(p=self.dropout))
            self.freq_new_fc = nn.Linear(feature_dim, num_class)

        std = 0.001
        if self.freq_new_fc is None:
            normal_(getattr(self.freq_model, self.freq_model.last_layer_name).weight, 0, std)
            constant_(getattr(self.freq_model, self.freq_model.last_layer_name).bias, 0)
        else:
            if hasattr(self.freq_new_fc, 'weight'):
                normal_(self.freq_new_fc.weight, 0, std)
                constant_(self.freq_new_fc.bias, 0)

    def _prepare_base_model(self, base_model):
        print('=> base model: {}'.format(base_model))

        self.base_model = getattr(torchvision.models, base_model)(True if self.pretrain == 'imagenet' else False)

        self.base_model.last_layer_name = 'fc'
        self.input_size = 224
        self.input_mean = [0.485, 0.456, 0.406]
        self.input_std = [0.229, 0.224, 0.225]

        self.base_model.avgpool = nn.AdaptiveAvgPool2d(1)

    def _prepare_freq_model(self, base_model):
        print('=> freq model: {}'.format(base_model))

        self.freq_model = getattr(torchvision.models, base_model)(False)

        self.freq_model.last_layer_name = 'fc'

        self.freq_model.avgpool = nn.AdaptiveAvgPool2d(1)

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(TSN, self).train(mode)
        count = 0
        if self._enable_pbn and mode:
            print("Freezing BatchNorm2D except the first one.")
            for m in self.base_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    count += 1
                    if count >= (2 if self._enable_pbn else 1):
                        m.eval()
                        # shutdown update in frozen mode
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False

    def partialBN(self, enable):
        self._enable_pbn = enable

    def forward(self, input, epoch, warm_up_epoch):
        if self.two_stream:
            ######## Frequency domain stream ########
            batch_size = input.shape[0]
            height = input.shape[2]
            width = input.shape[3]

            # minikinetics
            freqwise_mean = torch.tensor([[3.11999440e+02, 2.92734101e+02, 2.73512572e+02],
                                          [1.31976900e-01, 4.65869393e-03, 1.07130773e-01],
                                          [-7.42678419e-01, -8.82907354e-01, -8.00437933e-01],
                                          [1.45772034e-01, 1.69715159e-01, 1.56224231e-01],
                                          [1.94490182e-02, 3.83501204e-02, 3.46578196e-02],
                                          [-1.36656539e-02, -1.99214745e-02, -2.45398486e-02],
                                          [-1.46741678e-01, -1.39614321e-01, -1.30619307e-01],
                                          [5.87596261e-02, 6.39000697e-02, 6.71322088e-02]]).cuda()
            freqwise_std = torch.tensor([[99.39481043, 99.41126767, 105.53474589],
                                         [23.90491416, 22.84780224, 23.67578776],
                                         [16.03833016, 15.46310555, 16.23684146],
                                         [12.05854863, 11.69103394, 12.07319077],
                                         [9.48843771, 9.19222063, 9.49688663],
                                         [7.61678615, 7.42603237, 7.62194055],
                                         [6.39031728, 6.23412457, 6.36752511],
                                         [5.77056411, 5.65715373, 5.81123869]]).cuda()

            # somethingv2
            # freqwise_mean = torch.tensor([[3.39679380e+02, 3.11332070e+02, 2.91804584e+02],
            #                               [1.14269173e+00, 1.00906878e+00, 7.75543490e-01],
            #                               [1.25870654e+00, 2.40422867e+00, 2.64837731e+00],
            #                               [5.98750330e-01, 7.14237337e-01, 6.52234057e-01],
            #                               [2.58246911e-01, 5.14654996e-01, 5.46017073e-01],
            #                               [1.01753115e-01, 1.61826005e-01, 1.31615562e-01],
            #                               [2.76338649e-02, 1.13862982e-01, 9.89421514e-02],
            #                               [-6.85346599e-03, 1.49679967e-02, 1.11790851e-02]]).cuda()
            # freqwise_std = torch.tensor([[100.81675686, 94.3826637, 96.1053402],
            #                              [19.46331317, 19.32941076, 20.23192787],
            #                              [12.79817142, 12.89036199, 13.42268755],
            #                              [8.98159843, 9.06062315, 9.40278515],
            #                              [6.69077387, 6.78101926, 7.00193444],
            #                              [5.16934646, 5.2024311, 5.34479011],
            #                              [4.23246954, 4.2268805, 4.31114941],
            #                              [3.77383492, 3.76747485, 3.83444023]]).cuda()

            # TODO: RGBRGBRGB... -> RRR...GGG...BBB...
            r_indices = (torch.arange(self.num_segments) * 3).cuda()
            input_r = torch.index_select(input.reshape(batch_size, self.num_segments * 3, -1), 1, r_indices)
            input_g = torch.index_select(input.reshape(batch_size, self.num_segments * 3, -1), 1, r_indices + 1)
            input_b = torch.index_select(input.reshape(batch_size, self.num_segments * 3, -1), 1, r_indices + 2)
            input_reshape = torch.cat((input_r, input_g, input_b), dim=1)

            # TODO: DCT
            input_DCT = []
            self.DCT_hat = self.DCT_hat.cuda()
            for batch in range(batch_size):
                input_DCT.append(torch.matmul(self.DCT_hat, input_reshape[batch, :, :]).unsqueeze(0))
            input_DCT = torch.cat(input_DCT, 0)

            # TODO: Frequency-wise normalization
            input_DCT_freqnorm = torch.zeros_like(input_DCT).cuda()
            input_DCT_freqnorm[:, : self.num_segments, :] = (input_DCT[:, : self.num_segments, :]
                                                             - freqwise_mean[:, 0].unsqueeze(0).unsqueeze(-1)) \
                                                            / freqwise_std[:, 0].unsqueeze(0).unsqueeze(-1)
            input_DCT_freqnorm[:, self.num_segments: 2 * self.num_segments, :] = (input_DCT[:,
                                                                                  self.num_segments: 2 * self.num_segments, :]
                                                                                  - freqwise_mean[:, 1].unsqueeze(0).unsqueeze(-1)) \
                                                                                 / freqwise_std[:, 1].unsqueeze(0).unsqueeze(-1)
            input_DCT_freqnorm[:, 2 * self.num_segments: 3 * self.num_segments, :] = (input_DCT[:,
                                                                                      2 * self.num_segments: 3 * self.num_segments, :]
                                                                                      - freqwise_mean[:, 2].unsqueeze(0).unsqueeze(-1)) \
                                                                                     / freqwise_std[:, 2].unsqueeze(0).unsqueeze(-1)

            # TODO: RRR...GGG...BBB... -> RGBRGBRGB...
            input_new = torch.zeros_like(input_DCT_freqnorm).cuda()
            for i in range(3 * self.num_segments):
                if i < self.num_segments:
                    input_new[:, 3 * i, :] = input_DCT_freqnorm[:, i, :]
                elif i < 2 * self.num_segments:
                    input_new[:, 3 * (i - self.num_segments) + 1, :] = input_DCT_freqnorm[:, i, :]
                else:
                    input_new[:, 3 * (i - 2 * self.num_segments) + 2, :] = input_DCT_freqnorm[:, i, :]

            # input_new = torch.zeros_like(input_DCT).cuda()
            # for i in range(3 * self.num_segments):
            #     if i < self.num_segments:
            #         input_new[:, 3 * i, :] = input_DCT[:, i, :]
            #     elif i < 2 * self.num_segments:
            #         input_new[:, 3 * (i - self.num_segments) + 1, :] = input_DCT[:, i, :]
            #     else:
            #         input_new[:, 3 * (i - 2 * self.num_segments) + 2, :] = input_DCT[:, i, :]

            ######## Time domain stream ########
            if epoch == None or epoch >= warm_up_epoch:
                # TODO: normalization
                input_norm = input / 255
                rep_mean = self.input_mean * self.num_segments
                rep_std = self.input_std * self.num_segments
                for i in range(batch_size):
                    for j, m, s in zip(input_norm[i], rep_mean, rep_std):
                        j.sub_(m).div_(s)

            ######## forward ########
            freq_out = self.freq_model(input_new.reshape(batch_size * self.num_segments, 3, height, width))

            if self.dropout > 0:
                freq_out = self.freq_new_fc(freq_out)

            freq_out = freq_out.reshape(batch_size, self.num_segments, -1)
            freq_out = self.consensus(freq_out)

            if epoch == None or epoch >= warm_up_epoch:
                base_out = self.base_model(input_norm.view((-1, 3) + input_norm.size()[-2:]))
                if self.dropout > 0:
                    base_out = self.new_fc(base_out)
                base_out = base_out.view((-1, self.num_segments) + base_out.size()[1:])
                output = self.consensus(base_out)
                return (output.squeeze().softmax(dim=1) + freq_out.squeeze().softmax(dim=1)) / 2
            else:
                return freq_out.squeeze()

        else:
            base_out = self.base_model(input.view((-1, 3) + input.size()[-2:]))

            if self.dropout > 0:
                base_out = self.new_fc(base_out)

            if not self.before_softmax:
                base_out = self.softmax(base_out)

            base_out = base_out.view((-1, self.num_segments) + base_out.size()[1:])
            output = self.consensus(base_out)
            return output.squeeze()

    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * 256 // 224

    def get_augmentation(self, flip=True):
        if flip:
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]),
                                                   GroupRandomHorizontalFlip(is_flow=False)])
        else:
            print('#' * 20, 'NO FLIP!!!')
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66])])