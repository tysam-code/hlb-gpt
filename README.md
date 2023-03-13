[![Twitter URL](https://img.shields.io/twitter/url/https/twitter.com/hi_tysam.svg?style=social&label=Follow%20%40TySam_And)](https://twitter.com/hi_tysam) [![Support me on Patreon](https://img.shields.io/endpoint.svg?url=https%3A%2F%2Fshieldsio-patreon.vercel.app%2Fapi%3Fusername%3Dtysam%26type%3Dpatrons%26suffix%3Dsponsors&style=flat)](https://patreon.com/tysam)

## hlb-GPT
Welcome to the hyperlightspeedbench-gpt (hlb-gpt) repo! :D

### How to Run


`git clone https://github.com/tysam-code/hlb-gpt && cd hlb-gpt && python -m pip install -r requirements.txt && python main.py`


If you're curious, this code is generally Colab friendly (in fact -- most of this was developed in Colab!). Just be sure to uncomment the reset block at the top of the code.


### Main

Goals:
* minimalistic
* hackable
* beginner-friendly
* torch- and python-idiomatic
* reference implementation for performant applied ML
* few external dependencies (currently only torch, torchvision, and tiktoken)
* eventual world-record-speed single-GPU training times on at least one LLM benchmark (currently via the A100! :D)


This code implements a fast-training language model baseline that achieves under ~3.8 val loss (44.7 perplexity) on WikiText-103 in just over 3 minutes. It is currently a relatively minimal model based on relatively faithful reimplementation of a basic GPT language model as defined in Karpathy's excellent [nanoGPT](https://github.com/karpathy/nanoGPT) repository**. We've made a number of changes to improve the training speed on the tiny (~100M) training set that we use. For the rationale behind the 3.8 val loss target, please see the '[Why 3.8 Val Loss?](#why-3.8-val-loss?)' section.


This is a very focused codebase which attempts to maximize code understandability and minimize code length. At the same time, it aims to be very hackable to let people test out new ideas and get back initial results rapidly. We also want to keep this code as accessible as possible to a wide range of users and usecases -- mainly through layout, helpful code comments, and simplicity. We also only target one piece of hardware -- the A100 currently -- but attempt to maintain accessibility by providing options for people with less GPU memory. As as result of all of this, this means that this implementation really doesn't have much in the way of fancy features. It downloads and loads the data, creates the network, runs the training loop and that's about it -- if you want anything on top of that, it should be easily to implement with how modular the code is. That said, feel free to open an issue if there is something critical that I've missed!


Finally, this code is meant to be fast -- as fast as possible. Please keep an eye out for further training speedups in future updates, since this is just the baseline after all. This code is in a single file and extremely flat, but is not as durable for long-term production-level bug maintenance. You're meant to check out a fresh repo whenever you have a new idea. Part of the recommended workflow at scale is that if you're at an organization that needs a modified 'base repo', to modify this repo and use that as your new base internally. I oftentimes use a branching tree structure several repos deep in my work and I find it to be a great way to rapidly explore/context switch/roll back between different problem-solving domains. It's also another reason why I keep the base repo so simple.


Speaking of which -- I believe that this repo is excellent for rapid idea exploration -- almost everywhere in the pipeline is exposed and built to be hackable/user-friendly for very fast, iterated research. I hope you enjoy using it! :D Please let me know if you have any feedback. I hope to continue publishing updates to this in the future, so your support and thoughts on what worked for you/what didn't is encouraged. Share this repo with someone you know that might like it! I can't wait to see what you come up with. :) :D


Feel free to check out my [Patreon](https://www.patreon.com/user/posts?u=83632131) if you like what I'm doing here and want to support me in my work! Any funding for GPU hours is greatly appreciated! :)))) Additionally, if you want me to work up to a part-time amount of hours with you via short contract work or otherwise, feel free to reach out to me at hire.tysam@gmail.com. I'd love to hear from you.



** I also recommend checking out the [GPT video](https://youtu.be/kCc8FmEb1nY) from his excellent zero to hero series if you are generally unfamiliar with large language models (LLMs) :) :D

### Known Bugs / Potential Problem Areas

The Colab-specific code is commented out at the top, and some of the model weight initialization and flops/mfu/etc calculations might require you to update them manually if you are making significant changes to the network. There's currently some bugs relating to the dataloader and the number of steps we run doing training -- if just manually measuring the time to get to <3.8 loss, you should be okay for now. Hopefully I'll be able to fix this in a future release.

### Why 3.8 Val Loss?

Basically -- here we have very similar reasoning to [hlb-CIFAR10](https://github.com/tysam-code/hlb-CIFAR10#why-a-convnet-still-why-cifar10-arent-transformers-the-new-thing-now) -- that is, the information we gain from an experiment (for the most part) should follow the same trend(s) as longer-running or larger scale experiments, albeit with some more noise. This isn't always true, but I've personally found it to hold true more than most people generally seem to expect it to. It does vary on a case-by case basis, but even for the edge cases, there is a good short-term analog that we can use. On the opposite hand, if we do longer/larger runs for most of our experiments, we can lose a lot of time, brainpower, unbiased experimenting, and the cost of context switching (!!!). Really short experiment cycles allow us to run more experiments and cover more exploratory ground, which gives us a broader exposure to the general shape of the problem than slow, methodical runs. It also allows us to reduce our bias and introduce some variance to the research process, because we don't always have to plan as hard before each run due to how short each experiment is. This introduces some noise into the experimentation process, which I've found to be a surprisingly important component for overcoming human bias and leading to new discoveries.


There's a lot more to it than that, but especially in the realm of generative pretraining -- for the 'real world usecase', in the long runs, we're ideally/hopefully going to only be seeing each token with the full network only _once_. This means that for this problem, we basically have an open field where we just have to (hopefully) maximize the rate of information acquisition into the network as fast as we possibly can. A val loss of 3.8 is right around where the training curves start flattening out during early training, but it still requires the network to learn some temporal and topical dependencies to get to that point. Values like 3.6 or even 3.4 can work well, but they seem to be in the realm of diminishing returns, especially for initial exploration. If we get fast enough and we find that we should move the marker for some reason at some point in the future, then that's certainly possible! But for now, 3.8 (roughly) seems to be a good enough signal but without too much compromise in training time. For training taking just over ~6 minutes or so, the output from the networks is not too bad (provided the temperature is low enough)!


One other bonus would be that for where we're at in the training process, a val loss of 3.8 I believe is just slightly over 1 single training pass over the data -- so hopefully with some optimization we can get there with only 1 training pass over the data, better simulating what we'd be doing with a larger dataset.

### Citation

If you use this work in your research, please cite
`@software{Balsam_hlb-gpt_2023,
   author={Balsam, Tysam&},
   month={3},
   title={{hlb-gpt}},
   url={https://github.com/tysam-code/hlb-gpt},
   version = {0.1.0},
   year = {2023}}`

### Bugs & Etc.

If you find a bug, open an issue! L:D If you have a success story, let me know! It helps me understand what works and doesn't more than you might expect -- if I know how this is specifically helping people, that can help me further improve as a developer, as I can keep that in mind when developing other software for people in the future. :D :)