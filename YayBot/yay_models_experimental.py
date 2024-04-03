from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.torch_utils import FLOAT_MIN, FLOAT_MAX
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
from typing import Callable, Tuple
from collections import OrderedDict

# lot to do here, going to time box myself and chip away at it
# might make sense to do multiple different models rather than 1 super model with like tons of possible permutations

'''
working
    DONE figure out the split in init between discrete and continuous parts of the non_spatial features
    figure out how doing the embedding layer
        my notes say just to expand
        so I think I make the embedding lookup, make the row based on the index, then expand into the shape
            look over those torch embeddings from that kaggle example that you found and see if you can make sense of them
    then in forward figure out the concat and how shit is fed into the embeddings
    then review everything overall and see what I am missing

luxai net. inspired by https://github.com/IsaiahPressman/Kaggle_Lux_AI_2021
#https://github.com/IsaiahPressman/Kaggle_Lux_AI_2021/blob/main/lux_ai/nns/conv_blocks.py
all features become spatial, then many resblocks over it
key parts
    TEST res blocks with adjustable number
    1x1 Network in network
        happens in three places:
            TEST continuous: concat - 1x1 conv2d then leakyrelu
            discrete: concat - 1x1 conv2d then leakyrelu
            TEST after everything has been concat'd and before the CNN begins
    embeddings for non spatial
    TEST squeeze excitations (no other normalization)(
    different paths for continuous and non continuous
my adjustments
    TEST outputting a VF and action head (lux env is a bit trickier and he outputs a clever thing customized for that env)
    
to do
    need to design the wrapper to make the various dict parts
    in init and forward need to do the various non-spatial part things

later/unclear
for continuous non-spatial, not sure if i'm getting the shape correct prior to the 1x1
for the merged shit, unclear if i'm getting the shape correct before the 1x1
unclear when I should extend vs. append in the init
dilation and stride params for final model
    
'''

# for init layers. used in IMPALANet by cleanRL so using in some of my networks here
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class LuxResidualBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            height: int,
            width: int,
            kernel_size: int = 3,
            normalize: bool = False,
            activation: Callable = nn.ReLU,
            squeeze_excitation: bool = True,
            rescale_se_input: bool = True,
            **conv2d_kwargs
    ):
        super(LuxResidualBlock, self).__init__()

        # Calculate "same" padding
        # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        # https://www.wolframalpha.com/input/?i=i%3D%28i%2B2x-k-%28k-1%29%28d-1%29%2Fs%29+%2B+1&assumption=%22i%22+-%3E+%22Variable%22
        assert "padding" not in conv2d_kwargs.keys()
        k = kernel_size
        d = conv2d_kwargs.get("dilation", 1)
        s = conv2d_kwargs.get("stride", 1)
        padding = (k - 1) * (d + s - 1) / (2 * s)
        assert padding == int(padding), f"padding should be an integer, was {padding:.2f}"
        padding = int(padding)

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(kernel_size, kernel_size),
            padding=(padding, padding),
            **conv2d_kwargs
        )
        # We use LayerNorm here since the size of the input "images" may vary based on the board size
        self.norm1 = nn.LayerNorm([in_channels, height, width]) if normalize else nn.Identity()
        self.act1 = activation()

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(kernel_size, kernel_size),
            padding=(padding, padding),
            **conv2d_kwargs
        )
        self.norm2 = nn.LayerNorm([in_channels, height, width]) if normalize else nn.Identity()
        self.final_act = activation()

        if in_channels != out_channels:
            self.change_n_channels = nn.Conv2d(in_channels, out_channels, (1, 1))
        else:
            self.change_n_channels = nn.Identity()

        if squeeze_excitation:
            self.squeeze_excitation = SELayer(out_channels, rescale_se_input)
        else:
            self.squeeze_excitation = nn.Identity()

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        x, input_mask = x
        identity = x
        x = self.conv1(x) * input_mask
        x = self.act1(self.norm1(x))
        x = self.conv2(x) * input_mask
        x = self.squeeze_excitation(self.norm2(x), input_mask)
        x = x + self.change_n_channels(identity)
        return self.final_act(x) * input_mask, input_mask


class LuxNet(TorchModelV2, nn.Module):
    # https://github.com/AIcrowd/neurips2020-procgen-starter-kit/blob/master/models/impala_cnn_torch.py
    # https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_procgen.py
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name) #, **kwargs I don't think ray allows these kwargs here
        nn.Module.__init__(self)

        orig_space = getattr(obs_space, "original_space", obs_space)
        assert (
                isinstance(orig_space, gym.spaces.Dict)
                and "action_mask" in orig_space.spaces
                and "spatial" in orig_space.spaces
                #and "non_spatial" in orig_space.spaces
                #unclear how doing non-spatial in this point at this time
        )

        spatial_channels, self.spatial_height, self.spatial_width = orig_space[
            "spatial"].shape  # channels, height, width of the board

        # i'm thinking I make an activation dict and a layer dict here with torch.nn.module
        # then in forward grab them all and concat

        # non spatial obs are treated differently for continuous vs. non-continuous
        # non spatial continous obs are broadcast into board shape, concatenated, given a 1x1 conv2d, then leaky relu
        # add keys to an ordered dict so concat them in the correct order down below
        self.non_spatial_continuous_keys_dict = OrderedDict()
        non_spatial_continuous_space_embedding_layers = []
        non_spatial_discrete_space_embedding_layers = []
        non_spatial_continuous_channels = 0
        non_spatial_discrete_channels = 0
        non_spatial_cont_check = "non_spatial_continuous"
        space_names = list(orig_space.spaces.keys())
        space_names.sort()
        for sn in space_names:
            if non_spatial_cont_check in sn:
                self.non_spatial_continuous_keys_dict[sn] = True
                non_spatial_continuous_channels += 1
            else:
                self.non_spatial_continuous_keys_dict[sn] = False
                non_spatial_discrete_channels += 1

        # non-spatial continuous: concat 1x1 then LeakyRelu
        non_spatial_continuous_space_embedding_layers.extend([
            nn.Conv2d(non_spatial_continuous_channels, non_spatial_continuous_channels, (1, 1)),
            nn.LeakyReLU()
        ])
        self.non_spatial_continuous_space_embedding = nn.Sequential(*non_spatial_continuous_space_embedding_layers)

        # non-spatial discrete: embedding layers, concat, 1x1 Conv2D then leaky relu
        non_spatial_discrete_space_embedding_layers.extend([
            nn.Conv2d(non_spatial_discrete_channels, non_spatial_discrete_channels, (1, 1)),
            nn.LeakyReLU()
        ])
        self.non_spatial_discrete_space_embedding = nn.Sequential(*non_spatial_discrete_embedding_layers)

        # after the spatial, non-spatial contin and non-spatial discreted are concatted there is a sequential layer then a 1x1 conv2d
        # I am uncertain how to do this
        total_channels = spatial_channels + non_spatial_channels
        merger_layers = []
        # merger_layers.extend([
        #     nn.Conv2d(total_channels, total_channels, (1, 1)),
        #     nn.LeakyReLU()
        # ])
        merger_layers.append(nn.Conv2d(total_channels, total_channels, (1, 1)))
        self.merger = nn.Sequential(*merger_layers)

        res_block_layers = kwargs.get("num_res_layers", 8) # first version had 8, final version had 24
        hidden_nodes_head = kwargs.get("hidden_nodes_head", 256)  # first version had 8, final version had 24

        # making the residual block part of the network
        res_seqs = []
        for _ in res_block_layers:
            # using default settings of Lux, could add argumetns to modify this
            res_seqs.append(LuxResidualBlock(
                in_channels=total_channels, # official uses 128 but I think i'm okay using the num channels
                out_channels=total_channels,
                height=self.spatial_height,
                width=self.spatial_width,
                kernel_size=5, # official uses 5 but maybe 3 would be better here since smaller image size?
                normalize=False,
                activation=nn.LeakyReLU,
                rescale_se_input=True,
            ))

        # luxnet actual has a really inspired head. here just doing the standard flatten, NN layer, and then a policy output and value function output
        # flatten and one hidden layer. using leakyrelu because rest of luxnet does but unclear on optimal size and activation
        res_seqs += [
            nn.Flatten(),
            nn.LeakyReLU(),
            layer_init(nn.Linear(in_features=spatial_channels * self.spatial_height * self.spatial_width, out_features=hidden_nodes_head)),
            nn.LeakyReLU(),
        ]
        self.network = nn.Sequential(*res_seqs)

        # init output in forward and then put critic over this in value_function
        self._output = None
        # actor is the policy head, critic is the value function head
        self.actor = layer_init(nn.Linear(hidden_nodes_head, action_space.n), std=0.01)  # traditional impalanet is 256
        self.critic = layer_init(nn.Linear(hidden_nodes_head, 1), std=1)  # traditional impalanet 256 hidden nodes

        # base_model = nn.Sequential(
        #     conv_embedding_input_layer,
        #     *[ResidualBlock(
        #         in_channels=flags.hidden_dim,
        #         out_channels=flags.hidden_dim,
        #         height=MAX_BOARD_SIZE[0],
        #         width=MAX_BOARD_SIZE[1],
        #         kernel_size=flags.kernel_size,
        #         normalize=flags.normalize,
        #         activation=nn.LeakyReLU,
        #         rescale_se_input=flags.rescale_se_input,
        #     ) for _ in range(flags.n_blocks)]
        # )

    def forward(self, input_dict, state, seq_lens):
        action_mask = input_dict["obs"]["action_mask"]
        spatial_obs = input_dict["obs"]["spatial"]
        # TBD non_spatial_input = input_dict["obs"]["non_spatial"]

        # non spatial continuous obs:
        #TBD take the various inputs, treat and concat them
        non_spatial_continuous_outs = []
        for sn in self.non_spatial_continuous_keys_dict.keys():
            non_spatial_continous_x = input_dict["obs"][sn]
            # Broadacast to the correct shape. yeah I'm pretty unclear about this
            # I think it comes in as batch_size as dim 0, then a single value so 2 dimensional tensor (batch_dim,1)
            # I NEED TO TEST THIS
            non_spatial_continous_x = non_spatial_continous_x.unsqueeze(-1) # I need to do unsqueeze twice if the shape is (batch_dim,) non_spatial_continous_x.unsqueeze(-1).unsqueeze(-1)
            non_spatial_continous_x = non_spatial_continous_x.expand(-1, self.spatial_height, self.spatial_width)
            non_spatial_continuous_outs.append(non_spatial_continous_x)

        # concat the non-spatial continous toggether
        non_spatial_continuous_outs_combined = self.non_spatial_continuous_space_embedding(torch.cat(non_spatial_continuous_outs, dim=1))

        # spatial continuous obs

        # concat the spatial and the two non-spatial types together
        merged_outs = self.merger(torch.cat([spatial_obs, non_spatial_continuous_outs_combined, non_spatial_discrete_outs_combined], dim=1))
        # send the concat'd inputs into the CNN
        merged_outs = self.network(merged_outs)
        self._output = merged_outs# used for VF/critic head
        logits = self.actor(merged_outs) # policy head

        # apply masks to actions: ie make non-feasible actions as small as possible
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN, max=FLOAT_MAX)
        masked_logits = logits + inf_mask

        return masked_logits, state

    def value_function(self):
        assert self._output is not None, "must call forward first!"
        return torch.reshape(self.critic(self._output), [-1])


# used after 2nd conv layer activation and norm
class SELayer(nn.Module):
    def __init__(self, n_channels: int, rescale_input: bool, reduction: int = 16):
        super(SELayer, self).__init__()
        self.rescale_input = rescale_input
        self.fc = nn.Sequential(
            nn.Linear(n_channels, n_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(n_channels // reduction, n_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, input_mask: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        # Average feature planes
        if self.rescale_input:
            y = torch.flatten(x, start_dim=-2, end_dim=-1).sum(dim=-1)
            y = y / torch.flatten(input_mask, start_dim=-2, end_dim=-1).sum(dim=-1)
        else:
            y = torch.flatten(x, start_dim=-2, end_dim=-1).mean(dim=-1)
        y = self.fc(y.view(b, c)).view(b, c, 1, 1)
        return x * y.expand_as(x)




# for impala net custom
# taken from https://github.com/AIcrowd/neurips2020-procgen-starter-kit/blob/142d09586d2272a17f44481a115c4bd817cf6a94/models/impala_cnn_torch.py
class ResidualBlockCustom(nn.Module):
    def __init__(self, channels, kernel_size=3, activation: Callable = nn.ReLU, use_batch_norm=False, use_squeeze_e=False):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, padding=1)
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, padding=1)

        if use_batch_norm:
            self.bn_1 = nn.BatchNorm2d(channels)
            self.bn_2 = nn.BatchNorm2d(channels)
        else:
            self.bn_1 = nn.Identity()
            self.bn_2 = nn.Identity()

        self.activation_1 = activation()
        self.activation_2 = activation()

        if use_squeeze_e:
            self.squeeze_excitation = SELayer(channels, rescale_se_input=False) # unclear if rescale applies here
        else:
            self.squeeze_excitation = nn.Identity()

    def forward(self, x):
        inputs = x
        x = self.activation_1(x)
        x = self.conv0(x)
        x = self.bn_1(x)
        x = self.activation_2(x)
        x = self.conv1(x)
        x = self.squeeze_excitation(self.bn_2(x))
        return x + inputs


# for impalanet custom
class ConvSequenceCustom(nn.Module):
    def __init__(self, input_shape, out_channels, kernel_size=3, activation: Callable = nn.ReLU, use_batch_norm=False,
                 use_squeeze_e=False):
        super().__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels
        self.conv = nn.Conv2d(in_channels=self._input_shape[0], out_channels=self._out_channels, kernel_size=kernel_size, padding=1)
        self.res_block0 = ResidualBlockCustom(self._out_channels, kernel_size, activation, use_batch_norm, use_squeeze_e)
        self.res_block1 = ResidualBlockCustom(self._out_channels, kernel_size, activation, use_batch_norm, use_squeeze_e)

    def forward(self, x):
        x = self.conv(x)
        x = nn.functional.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.res_block0(x)
        x = self.res_block1(x)
        assert x.shape[1:] == self.get_output_shape()
        return x

    def get_output_shape(self):
        _c, h, w = self._input_shape
        return (self._out_channels, (h + 1) // 2, (w + 1) // 2)





class IMPALANetCustom(TorchModelV2, nn.Module):
    # IMPALANet with custom options
    # https://github.com/AIcrowd/neurips2020-procgen-starter-kit/blob/master/models/impala_cnn_torch.py
    # https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_procgen.py
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # are the non spatial features inputted as a collection of dicts or just an array
        self.is_non_spatial_dict = kwargs.get("is_non_spatial_dict", False)
        self.is_embeddings = kwargs.get("is_embeddings", False)  # embeddings for non-spatial features
        # self.embed_feat

        # read non_spatial features from dict option
        # assert to check that dictionary keys are there
            # list of embeddings
                # with embedding dims
            # list of non embeddings
        # dict of embedding dims
        # get list of features to embed from kwargs
        # get list of non features to embed
        # in init
            # create a nn.ModuleDict
                # for the embeddings
                # set up the embeddings
                # set up the layers for the embedding
                # set up the activations for the embeddings
            # track length of non-spatial non embeddings
                # concat layer joining that plus length of embeddings
                # then activation off of that
                # then arg for how many additional layers to have
            # want an option for embedding to spatial and non spatial
                # non spatial is above
                # if to spatial, have a non embedding path:
                    # basically just broadcast for each and scale by max value (pass max value in dict)
            # arg for number of layers post flatten of spatial
                # will sometimes be pure spatial (since non spatial join at beginning of res block) and sometimes concat, make sure it works both ways (this is in forward)

        # in forward

        # the one by one stuff
            # REVIEW init code for non 1x1
            # init code for 1x1
                # do the embeddings
                # get the sizes
            # forward code for 1x1
            # forward code for non 1x1
            # doesn't matter if embeddings or not, this happens for all non-spatial
            # happens in two places: non spatial discrete and non spatial continuous (and non-embed)
            # always happens after the concat, then join then concat

        # self.non_spatial_module_dict = nn.ModuleDict({}) #https://pytorch.org/docs/stable/generated/torch.nn.ModuleDict.html




        orig_space = getattr(obs_space, "original_space", obs_space)
        assert (
                isinstance(orig_space, gym.spaces.Dict)
                and "action_mask" in orig_space.spaces
                and "spatial" in orig_space.spaces
                and "non_spatial" in orig_space.spaces
        )
        # customize the CNN
        kernel_size = kwargs.get("kernel_size", 3) # can be int or tuple
        channel_sequences = kwargs.get("channel_sequences", [16, 32, 32])
        activation_function_cnn = kwargs.get("activation_function_cnn", nn.ReLU)  # alternative leaky relu
        self.spatial_flatten_activation = activation_function_cnn()  # activation function after the flatten
        is_batch_norm_cnn = kwargs.get("is_batch_norm", False)  # batch norm in various parts
        is_squeeze_e = kwargs.get("is_squeeze_e", False)  # squeeze excitation

        # customize FC net
        hidden_nodes = kwargs.get("hidden_nodes", 256)
        agent_layer_hidden_nodes = kwargs.get("agent_layer_hidden_nodes", 256) # this can easily eat up millions of parameters if set higher
        # activation function for non-spatial and any layers after combined with spatial
        activation_function_fc = kwargs.get("activation_function_fc", nn.ReLU)  # alternative leaky relu
        self.non_spatial_activation = activation_function_fc()
        self.concat_activation = activation_function_fc()



        self.is_non_spatial_fcnn = kwargs.get("is_non_spatial_fcnn", False) # fully connected NN for subsets of non-spatial features before concat
        self.is_1v1_cnn = kwargs.get("is_1v1_cnn", False) # network in network for non-spatial

        # (44, 17, 28) is spatial shape. 44 channels 17 height, 28 width
        c, h, w = orig_space["spatial"].shape
        if self.is_non_spatial_to_2D:
            c += non_spatial_channels # to do figure this out
        shape = (c, h, w)
        conv_seqs = []
        for out_channels in channel_sequences:#[16, 32, 32]:
            conv_seq = ConvSequence(shape, out_channels, kernel_size, activation_function_cnn, is_batch_norm_cnn,
                                    is_squeeze_e)
            shape = conv_seq.get_output_shape()
            conv_seqs.append(conv_seq)
        # doing my own flatten and concat because of non-spatial features
        # conv_seqs += [
        #     nn.Flatten(),
        #     nn.ReLU(),
        #     nn.Linear(in_features=shape[0] * shape[1] * shape[2], out_features=256),
        #     nn.ReLU(),
        # ]
        self.network = nn.Sequential(*conv_seqs)
        self.actor = layer_init(nn.Linear(agent_layer_hidden_nodes, action_space.n), std=0.01) # traditional impalanet 256
        self.critic = layer_init(nn.Linear(hidden_nodes, 1), std=1) # traditional impalanet 256 hidden nodes

        # Non-spatial input stream
        self.linear0 = layer_init(nn.Linear(orig_space['non_spatial'].shape[0], hidden_nodes))
        # layer for concatenate after the spatial and non spatial joins
        stream_size =  shape[0] * shape[1] * shape[2] #num outputs of flat spatial
        stream_size += hidden_nodes # num outputs of non-spatial
        self.linear1 = layer_init(nn.Linear(stream_size, hidden_nodes))
        # init output in forward and then put critic over this in value_function
        self._output = None

        # batch norm after spatial and after concat
        is_batch_norm_fc = kwargs.get("is_batch_norm", False)  # batch norm in various parts
        if is_batch_norm_fc:
            self.bn_1 = nn.BatchNorm1d(hidden_nodes)
            self.bn_2 = nn.BatchNorm1d(hidden_nodes)
        else:
            self.bn_1 = nn.Identity()
            self.bn_2 = nn.Identity()

        # concat the non spatial into the 2D channels then jsut use rez blocks
        self.is_non_spatial_to_2D = kwargs.get("is_non_spatial_to_2D", False)
        # optional argument to do a 1x1 convoluation on non-spatial features prior to them joining the spatial path
        self.is_1x1_non_spatial = kwargs.get("is_1x1_non_spatial", False)
        if self.is_non_spatial_to_2D:
            if self.is_1x1_non_spatial:
                # non-spatial features are in dictionaries
                # continuous/non embed
                    # broadcast to board size (in forward probably but confirm)
                    # concat
                    # 1x1 conv2d
                # then discrete
                    # embed, concat, 1x1 conv2d
                # concat continuous and discrete
                self.non_spatial_to_2D_cont_conv = nn.Conv2d(in_channels=TODOcontsize,
                                                        out_channels=TODOcontsize,
                                                        kernel_size=(1, 1))
                self.non_spatial_to_2D_cont_activation_function = nn.LeakyReLU()
                # to do discrete embeddings, will need to read up on how to do these and how to store best for forward to axise
                self.non_spatial_to_2D_discrete_activation_function = nn.LeakyReLU()
                self.non_spatial_to_2D_discrete_conv = nn.Conv2d(in_channels=TODOdiscsize,
                                                        out_channels=TODOdiscsize,
                                                        kernel_size=(1, 1))
                self.non_spatial_to_2D_conv = nn.Conv2d(in_channels=TODOdiscreteandcontinuoussize,
                                                        out_channels=TODOdiscreteandcontinuoussize,
                                                        kernel_size=(1, 1))
            else:
                # assumes no dictionary of features
                # broadcast the features in forward CONFIRM THIS
                # concat the features in forward
                # then go through 1x1 conv
                # then activation function
                if self.is_non_spatial_dict:

                    self.non_spatial_to_2D_conv = nn.Conv2d(in_channels=orig_space['non_spatial'].shape[0],
                                                            out_channels=orig_space['non_spatial'].shape[0],
                                                            kernel_size=(1,1))
                    self.non_spatial_to_2D_activation_function = activation_function_fc()
                else:
                    print("ERROR: non spatial featurs are in a dict")
                    assert 1 == 0

    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        action_mask = input_dict["obs"]["action_mask"]
        spatial_input = input_dict["obs"]["spatial"]
        non_spatial_input = input_dict["obs"]["non_spatial"]

        # spatial
        spatial_output = self.network(spatial_input)
        flatten_spatial = spatial_output.flatten(start_dim=1) # dim 0 is the batch sized, flatten after that
        flatten_spatial = self.spatial_flatten_activation(flatten_spatial) # IMPALANet procgen/cleanRL example relu after flatten
        #non spatial
        x2 = self.linear0(non_spatial_input)
        x2 = self.non_spatial_activation(self.bn_1(x2))
        flatten_x2 = x2.flatten(start_dim=1)
        # concat spatial and non-spatial, one more hidden layer before outputs (critic output done in vf function)
        concatenated = torch.cat((flatten_spatial, flatten_x2), dim=1)
        x3 = self.linear1(concatenated)
        self._output = self.concat_activation(self.bn_2(x3))

        # Output streams
        logits = self.actor(x3)

        # apply masks to actions: ie make non-feasible actions as small as possible
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN, max=FLOAT_MAX)
        masked_logits = logits + inf_mask

        return masked_logits, state

    def value_function(self):
        assert self._output is not None, "must call forward first!"
        #return torch.reshape(self.vf_layers(self._output), [-1])
        return torch.reshape(self.critic(self._output), [-1])