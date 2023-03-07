# Note: The main change we need to make if we're in Colab is to uncomment this below block (and to add !pip install tiktoken to a cell above).
# If we are in an ipython session or a notebook, clear the state to avoid bugs

# To enable torch.compile in the code, run the below code and reboot (note: this is risky as this can miss with your installs a bit)
#pip3 install numpy --pre torch[dynamo] --force-reinstall --extra-index-url https://download.pytorch.org/whl/nightly/cu117

"""
# don't forget these too:
# !pip3 install tiktoken
# !pip3 install numpy --pre torch[dynamo] --force-reinstall --extra-index-url https://download.pytorch.org/whl/nightly/cu117
try:
  _ = get_ipython().__class__.__name__
  ## we set -f below to avoid prompting the user before clearing the notebook state
  %reset -f
except NameError:
  pass ## we're still good
"""

import functools
from functools import partial
import urllib
import zipfile
import os

import torch
import torch.nn.functional as F
from torch import nn

# This seems like one of the best choices right now for a fast/lightweight/simple tokenizer.
import tiktoken

# Check if we're using pytorch 2 for those speedups
using_pytorch_2 = (int(torch.__version__.split('.')[0]) >= 2)

## <-- teaching comments
# <-- functional comments
# You can run 'sed -i.bak '/\#\#/d' ./main.py' to remove the teaching comments if they are in the way of your work. <3

# This can go either way in terms of actually being helpful when it comes to execution speed.
# torch.backends.cudnn.benchmark = True

# This code was built from the ground up to be directly hackable and to support rapid experimentation, which is something you might see
# reflected in what would otherwise seem to be odd design decisions. It also means that maybe some cleaning up is required before moving
# to production if you're going to use this code as such (such as breaking different section into unique files, etc). Think of this project
# as a digital breadboard for your research. :D :) That said, if there's ways this code could be improved and cleaned up, please do open a
# PR on the GitHub repo. Your support and help is much appreciated for this project! :)

# This is for testing that certain changes don't exceed X% portion of the reference GPU (here an A100)
# so we can help reduce a possibility that future releases don't take away the accessibility of this codebase.
#torch.cuda.set_per_process_memory_fraction(fraction=TBD./40., device=0) ## 40. GB is the maximum memory of the base A100 GPU

## Useful for debugging gradient issues (nan values and such). Warning -- its slow!
# torch.autograd.set_detect_anomaly(True)

# If we run out of memory, we can always increase the accumulate_steps and decrease this by the same factor (2x, 4x, etc).
# Currently, doing this doesn't result in _exact_ equivalence due to a quirk of implementation, but it should at some point in the future. 
batchsize = 64

# The default model here below is roughly ~29.94M parameters or so.
hyp = {
    'opt': {
        'lr': 2e-3,
        'weight_decay': 1e-3,
        'total_train_steps': 900,
        'eval_iter': 50, # how many train iterations we wait in between eval rounds (we don't include eval time in our performance stats) 
        'warmup_percent': .15, ## what percent of the training run to warmup the learning rate over
        'accumulate_steps': 5*(8//4), # via nanoGPT: simulate 8 gpus, 5 accum steps for a larger batchsize (esp necesary towards the end).
    },                                # since this model is tiny, we can increase the batchsize and decrease this by a factor of 4
    'net': {
        'residual_depth': 384, ## this should be a factor of 8 in some way to stay tensor core friendly
        'num_heads': 6,
        'num_blocks': 6,
    },
    'misc': {
        'num_tokens': 50304, # Rounded to the nearest value of 64 for dose sheeeeer speeeeeddssss :D
        'sequence_length': 256, # Very short sequence length, 
        'device': 'cuda',
        'dtype': torch.bfloat16,
        'data_location': 'data.pt',
    }
}

torch.backends.cuda.matmul.allow_tf32 = True # allows us to make sure we're able to use tensorfloat32 during training
torch.backends.cudnn.allow_tf32 = True
autocast_tensors = torch.amp.autocast(device_type=hyp['misc']['device'], dtype=hyp['misc']['dtype'])

#############################################
#                Dataloader                 #
#############################################

if not os.path.exists(hyp['misc']['data_location']):
    print("downloading data and tokenizing (1-2 min)")

    raw_data_source = 'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip'
    raw_data_cache = './data_raw/' # where to cache the data after downloading

    if not os.path.isfile(raw_data_cache):
        os.makedirs(raw_data_cache, exist_ok=True)
        urllib.request.urlretrieve(raw_data_source, f'{raw_data_cache}data.zip')

    with zipfile.ZipFile('data_raw/data.zip', 'r') as zip_ref:
        zip_ref.extractall('data_raw/')

    with open('data_raw/wikitext-103-raw/wiki.train.raw', 'r') as data_file:
        raw_train_data = data_file.read()

    with open('data_raw/wikitext-103-raw/wiki.valid.raw', 'r') as data_file:
        raw_eval_data = data_file.read()

    tokenizer = tiktoken.get_encoding("gpt2")
    raw_tokenized_train = tokenizer.encode_ordinary(raw_train_data)
    raw_tokenized_eval = tokenizer.encode_ordinary(raw_eval_data)

    train_tokenized = torch.tensor(raw_tokenized_train, device=hyp['misc']['device'], dtype=torch.int) # int64 is likely overkill for the amount of tokens we have...
    eval_tokenized = torch.tensor(raw_tokenized_eval, device=hyp['misc']['device'], dtype=torch.int)

    data = {
        'train': train_tokenized,
        'eval': eval_tokenized
        }

    torch.save(data, hyp['misc']['data_location'])
    print("completed the tokenization process!")

else:
    ## This is effectively instantaneous, and takes us practically straight to where the dataloader-loaded dataset would be. :)
    ## So as long as you run the above loading process once, and keep the file on the disc it's specified by default in the above
    ## hyp dictionary, then we should be good. :)
    data = torch.load(hyp['misc']['data_location'])


## As you'll note above and below, one difference is that we don't time loading the raw data to GPU since it's such a variable operation
## (this includes the tokenizing), and it can sort of get in the way of measuring other things.
#############################################
#            Network Components             #
#############################################

class LayerNorm(nn.Module):
    """ A variant of LayerNorm that lets us turn our weights and bias trainable values on and off with true and false, respectively"""
    def __init__(self, num_features, eps=1e-5, weight=True, bias=False):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features)) if weight else None
        self.bias = nn.Parameter(torch.zeros(num_features)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, weight=self.weight, bias=self.bias, eps=self.eps)


class AttentionBlock(nn.Module):
    """ A standard attention block (for now....?) """
    def __init__(self, num_features, sequence_length, num_heads):
        super().__init__()
        self.norm = LayerNorm(num_features, bias=False)
        # this function below is complicated and can be very tricky in instantiation and in calling, as it is implemented
        # strangely and the parameters are relatively unintuitive in how they are passed. before making any changes,
        # be sure to read https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html thoroughly
        self.attention = nn.MultiheadAttention(num_features, num_heads, bias=False, batch_first=True)
        
        ## this mask makes sure that each part of a sequence can only attend to the tokens that come behind it.
        self.causal_mask = torch.logical_not(torch.triu(torch.ones((sequence_length, sequence_length), device=hyp['misc']['device'], dtype=torch.bool))).T # TODO: way to simplify this?
    
    def forward(self, x):
        residual = x
        x = self.norm(x)
        ## https://youtu.be/kCc8FmEb1nY?t=3720 is a good explanation of self-attention
        x, _ = self.attention(x, x, x, attn_mask=self.causal_mask, need_weights=False)
        x = x + residual # haiku
        return x


class MLPBlock(nn.Module):
    """ A standard MLP block (for now....?) """
    def __init__(self, num_channels, expansion_factor=4):
        super().__init__()
        self.norm = LayerNorm(num_channels, bias=False)
        self.expand = nn.Linear(num_channels, num_channels*expansion_factor, bias=False)
        self.project = nn.Linear(expansion_factor*num_channels, num_channels, bias=False)
        self.activation = nn.GELU()
    
    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.expand(x)
        x = self.activation(x)
        x = self.project(x)
        x = x + residual # haiku
        return x


class PositionEmbedding(nn.Module):
    def __init__(self, sequence_length, num_features):
        super().__init__()
        ## This is a learnable embedding so that the network can tag features by how far back in time they are
        self.position_embedding = nn.Embedding(sequence_length, num_features)

    def forward(self, x):
        # You can customize this function as needed in your experiments. Have fun! :D
        range_lookup = torch.arange(x.shape[1], device=x.device)
        return self.position_embedding(range_lookup).unsqueeze(0) # unsqueeze since we only need one value for the entire batch dimension :)


#############################################
#          Init Helper Functions            #
#############################################

# Nothing here but us chickens! :D

#############################################
#            Network Definition             #
#############################################

class SpeedyLangNet(nn.Module):
    def __init__(self, network_dict):
        super().__init__()
        self.net_dict = network_dict # flexible, defined in the make_net function

    # This allows you to customize/change the execution order of the network as needed.
    def forward(self, x):
        x = self.net_dict['embedding'](x) + self.net_dict['position'](x) # We add the learned positional embeddings to the input embeddings in one shot
        for block in range(hyp['net']['num_blocks']):
            x = self.net_dict['attn_layers'][block](x) # note: residuals are included in the block definitions for these layers
            x = self.net_dict['mlp_layers'][block](x)  # note: residuals are included in the block definitions for these layers
        x = self.net_dict['norm'](x)
        x = self.net_dict['outputs'](x)
        return x


def make_net():
    # Note, you have to specify any arguments overlapping with defaults (i.e. everything but in/out depths) as kwargs so that they are properly overridden (TODO cleanup somehow?)
    network_dict = nn.ModuleDict({
        'embedding': nn.Embedding(hyp['misc']['num_tokens'], hyp['net']['residual_depth']),
        'position': PositionEmbedding(hyp['misc']['sequence_length'], hyp['net']['residual_depth']),
        'norm': LayerNorm(hyp['net']['residual_depth'], eps=1e-5, bias=False),
        'mlp_layers': nn.ModuleList([MLPBlock(hyp['net']['residual_depth']) for _ in range(hyp['net']['num_blocks'])]),
        'attn_layers': nn.ModuleList([AttentionBlock(hyp['net']['residual_depth'], hyp['misc']['sequence_length'], hyp['net']['num_heads']) for _ in range(hyp['net']['num_blocks'])]),
        'outputs': nn.Linear(hyp['net']['residual_depth'], hyp['misc']['num_tokens'], bias=False),
    })

    net = SpeedyLangNet(network_dict)
    net = net.to(hyp['misc']['device'])
    net.train()

    # Tie the input and output weights. Feel free to experiment with this!
    net.net_dict['embedding'].weight = net.net_dict['outputs'].weight

    for name, parameter in net.named_parameters():
        # TODO: Way to tidy this up for a future release?
        # Initialize both embedding layers (embedding and position) and the non-bias values of the 'normal' linear layers (outputs, expand, in_proj)
        if 'embedding' in name or 'position' in name or (('outputs' in name or 'expand' in name or 'in_proj' in name) and 'weight' in name):
            torch.nn.init.normal_(parameter.data, mean=0., std=.02) # normal init
        elif ((('project' in name and 'mlp' in name) or 'out_proj' in name or 'c_proj' in name) and 'weight' in name):
            # As noted in NanoGPT, this is from the GPT-2 paper. Also very similar from what I seee to the FixUp initialization for ResNets
            torch.nn.init.normal_(parameter.data, mean=0., std=.02/((2 * hyp['net']['num_blocks'])**.5)) # keeps variance from exploding when adding to the residual
        elif 'norm' not in name:
            print(f"warning, no initialization keyword match for: {name}!")

    # We compile the network later so that we can include compilation time inside of the training time to be an honest comparison against other methods.
    return net


def get_net_mfu_and_param_counts(net, batchsize, gradient_accumulation_steps, avg_time_per_batch):
    assert hyp['misc']['dtype'] in (torch.half, torch.bfloat16), "Flops calculation is inaccurate with types other than half-precision types"
    flops_dict = {}
    # The below is a very easy way to estimate total registered param counts. I believe the reason that we don't count the embedding-only layers by default
    # is because they function as a lookup table (so, technically not really FLOPs at all)
    params_dict = {
        name: parameter.numel() if 'position' not in name else 0
        for name, parameter in net.named_parameters()
    }
    total_num_params = sum(params_dict.values())

    # Rough flops estimate, see https://github.com/karpathy/nanoGPT/blob/ae3a8d5fdd3ddb8b13fab182723476523961e3ab/model.py#L327 for more info
    # Originally sourced from the PaLM paper, appendix B: https://arxiv.org/abs/2204.02311
    # This includes both the forward and backwards passes :D
    flops_for_single_input_token = 6 * total_num_params + 12 * hyp['net']['num_blocks'] * hyp['net']['residual_depth'] * hyp['misc']['sequence_length']
    flops_for_full_sequence = flops_for_single_input_token * hyp['misc']['sequence_length']
    flops_for_single_step = flops_for_full_sequence * batchsize * gradient_accumulation_steps

    current_flops_achieved = flops_for_single_step/avg_time_per_batch
    a100_total_possible_flops = 312e12 # TODO: is there a good way to make this more flexible? 
    mfu_value = current_flops_achieved / a100_total_possible_flops

    return mfu_value, params_dict


#############################################
#            Data Preprocessing             #
#############################################

# Nothing here for the baseline...but us chickens! :D

########################################
#          Training Helpers            #
########################################
@torch.no_grad()
def get_batches(data_dict, key, batchsize, sequence_length, num_steps):
    # All of the data here is on the GPU, so instead of permuting the underlying data, we just generate random (potentially non-unique) indices to sample.
    total_steps_in_iter = num_steps * batchsize
    shuffled = torch.randint(len(data_dict[key]) - sequence_length - 1, (batchsize*num_steps,), device=hyp['misc']['device'])

    # No augmentation for now (but maybe later?)
    tokens = data_dict[key].to(hyp['misc']['device'])

    for idx in range(num_steps):
        if not (idx+1)*batchsize > shuffled.shape[0]: ## Continue if there are tokens left to consume
            batch_starting_indexes = shuffled[idx*batchsize:(idx+1)*batchsize]
            batch_index_offsets = torch.arange(0, sequence_length, dtype=torch.long, device=hyp['misc']['device']) # Create offsets so we can index every item in the sequence
            batch_indexes = batch_starting_indexes.unsqueeze(-1) + batch_index_offsets.unsqueeze(0)

            # Unfortunately for now we have to flatten this since this seems to be the best way to sample it (then we have to reshape it again at the end)
            batch_indexes_flattened = batch_indexes.flatten()

            yield torch.take_along_dim(tokens, batch_indexes_flattened, dim=0).view(batchsize, sequence_length).long(), \
                torch.take_along_dim(tokens, batch_indexes_flattened+1, dim=0).view(batchsize, sequence_length).long()  # Returns each token as the input, then the _next_ token in the sequence as a target


def init_split_parameter_dictionaries(net):
    params_non_decay = {'params': [], 'lr': hyp['opt']['lr'], 'eps': 1e-8, 'betas': (.9, .95), 'weight_decay': 0.}
    params_decay     = {'params': [], 'lr': hyp['opt']['lr'], 'eps': 1e-8, 'betas': (.9, .95), 'weight_decay': hyp['opt']['weight_decay']}

    for name, p in net.named_parameters():
        if p.requires_grad:
            if 'outputs' in name or 'norm' in name or 'embedding' in name or 'position' in name or 'bias' in name:
                params_non_decay['params'].append(p)
            else:
                params_decay['params'].append(p) # Default to weight decay unless we explicitly specify that we don't want to use it

    return params_non_decay, params_decay


## Just your good ol', normal an' nice xentropy function. Which makes sense if (in the ideal scenario) we only see each datapoint one single time.
## However! If (esp for tiny datsets) we're seeing our data multiple times in a row, then maybe some smoothing to help regularize things a bit is in order.... :D
loss_fn = nn.CrossEntropyLoss(reduction='mean', ignore_index=-1)

logging_columns_list = ['epoch', 'current_steps', 'train_loss', 'val_loss', 'val_perplexity', 'train_acc', 'val_acc', 'a100_mfu', 'total_time_seconds']


# define the printing function and print the column heads
def print_training_details(columns_list, separator_left='|  ', separator_right='  ', final="|", column_heads_only=False, is_final_entry=False):
    print_string = ""
    if column_heads_only:
        for column_head_name in columns_list:
            print_string += separator_left + column_head_name + separator_right
        print_string += final
        print('-'*(len(print_string)))  # print the top bar
        print(print_string)
        print('-'*(len(print_string)))  # print the bottom bar
    else:
        for column_value in columns_list:
            print_string += separator_left + column_value + separator_right
        print_string += final
        print(print_string)
    if is_final_entry:
        print('-'*(len(print_string)))  # print the final output bar

print_training_details(logging_columns_list, column_heads_only=True)  ## print out the training column heads before we print the actual content for each run.

# We basically need to look up local variables by name so we can have the names, so we can pad to the proper column width.
## Printing stuff in the terminal can get tricky and in a previous life, this used an outside library. But some of the required stuff
## for that seemed even more heinous than this, unfortunately. So we switched to the "more simple" version of this!
format_for_table = lambda x, locals: (f"{locals[x]}".rjust(len(x))) \
                                          if x in locals and type(locals[x]) == int else "{:0.4f}".format(locals[x]).rjust(len(x)) \
                                      if x in locals and locals[x] is not None \
                                      else " "*len(x)


########################################
#           Train and Eval             #
########################################

def eval(net):
    ####################
    # Evaluation  Mode #
    ####################

    eval_batchsize = 64 # Note/semi-warning: This is slightly 'buggy' technically as it will drop the last batch in the eval set, but that should just add a bit of noise for our usecases
    num_steps = 24 # Do a slightly noisy fast eval (that should be good enough for our purposes)
    loss_list_val, acc_list = [], []

    with torch.no_grad():
        for inputs, targets in get_batches(data, key='eval', batchsize=eval_batchsize, sequence_length=hyp['misc']['sequence_length'], num_steps=num_steps):
            with autocast_tensors:
                outputs = net(inputs)
            val_loss = loss_fn(outputs.flatten(0, 1), targets.flatten(0, 1))
            loss_list_val.append(val_loss)
            acc_list.append((outputs.argmax(-1) == targets).float().mean())

        val_acc = torch.stack(acc_list).mean().item()
        val_loss = torch.stack(loss_list_val).mean().item()
        val_perplexity = 2.71828 ** val_loss

    return val_acc, val_loss, val_perplexity


def main():
    # Initializing constants for the whole run.
    total_time_seconds = 0.
    current_steps = 0

    num_steps_per_epoch = len(data['train']) // (batchsize * hyp['misc']['sequence_length'])
    # Note: This is a static calculation of the total number of microbatches up front, you may have to change this depending upon what you're tinkering with
    total_microbatch_steps = hyp['opt']['total_train_steps'] * hyp['opt']['accumulate_steps']

    # Get network
    net = make_net()

    ## Stowing the creation of these into a helper function to make things a bit more readable....
    params_non_decay, params_decay = init_split_parameter_dictionaries(net)
    adamw_speedup_mode = {'fused': True} if using_pytorch_2 else {'foreach': True}
    opt = torch.optim.AdamW([params_non_decay, params_decay], **adamw_speedup_mode)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=opt, max_lr=hyp['opt']['lr'], total_steps=hyp['opt']['total_train_steps'], pct_start=hyp['opt']['warmup_percent'], anneal_strategy='cos', cycle_momentum=False, div_factor=1e2, final_div_factor=.05)

    ## For accurately timing GPU code
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize() ## clean up any pre-net setup operations
    starter.record()

    # If you have pytorch 2.0, compiling the network before training can help us a ton
    # If you need to/are brave enough, check the top of this file for a pip command to install it.
    # It can bork your pre-existing setup though if you're not careful, so bewarb! D: :D
    if using_pytorch_2:
        net = torch.compile(net)

    #################
    # Training Mode #
    #################
    net.train()

    val_loss = None
    val_acc = None
    val_perplexity = None

    batch_iter_kwargs = {'data_dict': data, 'key': 'train', 'batchsize': batchsize, 'num_steps': total_microbatch_steps, 'sequence_length': hyp['misc']['sequence_length']}

    for microbatch_step, (inputs, targets) in enumerate(get_batches(**batch_iter_kwargs)):
        with autocast_tensors:
            outputs = net(inputs)

        loss = loss_fn(outputs.flatten(0, 1), targets.flatten(0, 1))

                # Quick non-eval summary every N training steps
        if (current_steps % 10 == 0
            and microbatch_step % hyp['opt']['accumulate_steps'] == 0
                and current_steps % hyp['opt']['eval_iter'] != 0):
            train_acc = (outputs.detach().argmax(-1) == targets).float().mean().item()
            train_loss = loss.detach().cpu().item()
            train_summary_variables = {'epoch': microbatch_step//num_steps_per_epoch, 'current_steps': current_steps, 'train_loss': train_loss, 'train_acc': train_acc}
            print_training_details(list(map(partial(format_for_table, locals=train_summary_variables), logging_columns_list)))

        loss.backward()

        ## Once we've accumulated steps over all of our microbatches, take a single full-batchsize step.
        if microbatch_step % hyp['opt']['accumulate_steps'] == 0:
            ## Step the optimizer, then scheduler
            opt.step()

            scheduler.step()

            ## Using 'set_to_none' I believe is slightly faster (albeit riskier w/ funky gradient update workflows) than under the default 'set to zero' method
            opt.zero_grad(set_to_none=True)
            current_steps += 1

            # Since we're not running over epochs anymore, we have to manually calculate what epoch it is.
            epoch = microbatch_step//num_steps_per_epoch

            if current_steps % hyp['opt']['eval_iter'] == 0:
                ender.record()
                torch.cuda.synchronize()
                total_time_seconds += 1e-3 * starter.elapsed_time(ender)
                train_loss = loss.detach().cpu().item() # To have an updated loss to compare with the eval loss

                opt.zero_grad(set_to_none=True)
                net.eval()

                val_acc, val_loss, val_perplexity = eval(net)
                average_time_per_batch = 1e-3 * starter.elapsed_time(ender)/hyp['opt']['eval_iter']

                a100_mfu, _ = get_net_mfu_and_param_counts(net, batchsize, hyp['opt']['accumulate_steps'], avg_time_per_batch=average_time_per_batch)
                is_final_eval = (current_steps == hyp['opt']['total_train_steps']) # If we're at the end of training, do a full eval instead

                # Print out our training details (sorry for the complexity, the whole logging business here is a bit of a hot mess once the columns need to be aligned and such....)
                ## We also check to see if we're on our final eval loop (assuming that max_num_steps lines up with the eval_iter value) so we can print the 'bottom' of the table for each round.
                print_training_details(list(map(partial(format_for_table, locals=locals()), logging_columns_list)), is_final_entry=is_final_eval)
                torch.cuda.synchronize()
                starter.record()
                net.train()  # Functionally shouldn't do anything with the base network, just adding this to guard against any bugs for any future changes that do require this <3 <3 <3

    return net.eval(), val_loss # Return the final validation loss achieved (not using the 'best validation loss' selection strategy, which I think is okay here....)


if __name__ == "__main__":
    val_loss_list = []
    for _ in range(5):
            _, val_loss = main()
            val_loss_list.append(val_loss)
    print(f"Average final val loss: {sum(val_loss_list)/len(val_loss_list)}")  # TODO add variance as well, later