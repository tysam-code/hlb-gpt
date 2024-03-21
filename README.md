[![Twitter URL](https://img.shields.io/twitter/url/https/twitter.com/hi_tysam.svg?style=social&label=Follow%20%40TySam_And)](https://twitter.com/hi_tysam) [![Support me on Patreon](https://img.shields.io/endpoint.svg?url=https%3A%2F%2Fshieldsio-patreon.vercel.app%2Fapi%3Fusername%3Dtysam%26type%3Dpatrons%26suffix%3Dsponsors&style=flat)](https://patreon.com/tysam)

## hlb-GPT

Welcome to the hyperlightspeedbench-gpt (hlb-gpt) repository! This project is meant to be the best tool for researchers wanting to quickly explore new LLM ideas. It is also intended to be a good starting point for new projects and new ML developers. The code is simple, performant and well-documented with good default hyperparameters. It also optionally scales from 46 M parameters (the default) to up to 3 B parameters on a single A100 just by changing the model_scale parameter -- the rest of the hyperparameters are automatically inferred (though the scaling feature is still in alpha as the large model hyperparameters still need tuning).

### How to Run


`git clone https://github.com/tysam-code/hlb-gpt && cd hlb-gpt && python -m pip install -r requirements.txt && python main.py`


This code was developed exclusively in Colab, but also runs in the terminal as well. If you are running it in Colab, be sure to uncomment the code block at the top.

### Main

This code achieves a ~3.80 validation loss on WikiText-103 within about 100 seconds or so on a single A100 with default settings. By default, it runs for 1000 steps before ending and running a demo inference on the trained network, though you can (and should!) change this value as you begin experimenting. The learning rate schedulers are set to run infinitely, the step count is just a cutoff. As one of the design decisions to keep things simple, this code does assume that you are using a 40 GB A100, though hopefully we will be able to port to more GPU memory sizes as the scaling rules solidify.

The code is very short -- just over 300 lines or so. It implements a number of novel (or at least, novel-to-the-author) concepts, including a LatentAttention block that efficiently fuses attention and the MLP blocks into one, learnable linear position embeddings to let the attention layers learn a dynamic attention length, a dynamic microbatch scheduler based upon the expected gradient norm, a specific set of parameter group schedules, and several other things of various potential novelty.

I originally referenced nanoGPT when originally writing this code, though this code has certainly become its own beast at this point! Much appreciation to Karpathy and contributors for that codebase.

One of the intents of this codebase is to minimize the time-to-result for a given experiment. My experience leads me to believe that this is a good thing to optimize for (my appreciation to Keller Jordan for conversations on this topic a little while back).

If you have any questions, please let me know. My Twitter DMs should be open, as well as my email.

### Contact

Much of this work is supported by a combination of being both self-funded and being funded from the support of people like you. My Patreon is at [Patreon](https://www.patreon.com/user/posts?u=83632131) if you like what I'm doing here and would like to see more work like this in the future. If you want me to work up to a part-time amount of hours with you via consulting or contract work, please feel free to reach out to me at hire.tysam@gmail.com. I'd love to hear from you.

### Citation

If you use this work in your research, please cite
`@software{hlb-gpt_2024,
   author={Fern},
   month={3},
   title={{hlb-gpt}},
   url={https://github.com/tysam-code/hlb-gpt},
   version = {0.4.0},
   year = {2024}}`
