# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

from torch import nn

from ops.basic_ops import ConsensusModule
from ops.transforms import *
from torch.nn.init import normal_, constant_

class TSN(nn.Module):
    def __init__(self, num_class, num_segments, modality,
                 base_model='resnet101', new_length=1,
                 consensus_type='avg', before_softmax=True,
                 dropout=0.5, img_feature_dim=256,
                 crop_num=1, partial_bn=True, print_spec=True, pretrain='imagenet',
                 fc_lr5=False):
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
            """.format(base_model, self.num_segments, self.dropout, self.img_feature_dim)))

        self._prepare_base_model(base_model)
        self._prepare_tsn(num_class)
        self.consensus = ConsensusModule(consensus_type)

        self.linear = nn.Linear(num_class, 1)

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

    def _prepare_base_model(self, base_model):
        print('=> base model: {}'.format(base_model))

        self.base_model = getattr(torchvision.models, base_model)(True if self.pretrain == 'imagenet' else False)

        self.base_model.last_layer_name = 'fc'
        self.input_size = 224
        self.input_mean = [0.485, 0.456, 0.406]
        self.input_std = [0.229, 0.224, 0.225]

        self.base_model.avgpool = nn.AdaptiveAvgPool2d(1)

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

    def forward(self, input):

        base_out = self.base_model(input.view((-1, 3) + input.size()[-2:]))

        if self.dropout > 0:
            base_out = self.new_fc(base_out)

        if not self.before_softmax:
            base_out = self.softmax(base_out)

        base_out = base_out.view((-1, self.num_segments * 3 // 2) + base_out.size()[1:])
        slow_out = base_out[:, : self.num_segments, :]

        def th_delete(tensor, indices):
            mask = torch.ones(tensor.numel(), dtype=torch.bool)
            mask[indices] = False
            return tensor[mask]
        base_ind = th_delete(torch.arange(self.num_segments * 3 // 2), torch.arange(self.num_segments // 2) * 2 + 1)
        base_ind = base_ind.cuda()
        base_out = torch.index_select(base_out, 1, base_ind)

        base_output = self.consensus(base_out)
        slow_output = self.consensus(slow_out)

        fast_output = self.linear(base_output.squeeze())
        slow_output = self.linear(slow_output.squeeze())

        return base_output.squeeze(), fast_output, slow_output

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