# ARC Challenge and My Approach
Introduction to the ARC Challenge
Artificial General Intelligence (AGI) research has reached an impasse, with current AI models struggling to adapt to new problems or create novel solutions without extensive training. The ARC (Abstraction and Reasoning Corpus) benchmark, introduced by François Chollet in 2019, aims to address this by measuring a system’s ability to acquire new skills and solve open-ended problems. Unlike traditional AI benchmarks that focus on specific skills, ARC evaluates general intelligence, making it a challenging yet crucial test for advancing AGI.

## The ARC Prize
To incentivize progress, the ARC Prize offers over $1,000,000 to anyone who can develop and open-source a solution that beats the ARC-AGI benchmark. This competition, hosted by Mike Knoop (Co-founder of Zapier) and François Chollet (Creator of ARC-AGI and Keras), invites innovators to break the current limitations of AI and contribute to the open-source community.

## My Approach
### Goal
My goal is to create a transformer capable of natively perceiving and generating data in the domain of integer grids, the format used in the ARC tasks.

### Strategy
#### Custom Tokenization and Embedding:

Existing transformers struggle with ARC tasks due to inadequate grid tokenization. Current tokenizers break grids into arbitrary characters, which is inefficient.

<div align="center">
        <img src="media/Example%20issues%20with%20existing%20tokenizers.png" alt="Example issues with existing tokenizers" width="400"/>
</div>

I will develop a simpler tokenizer and embedding space tailored for the integer grids. This approach will avoid the complexity of Vision Transformers' patching and tokenization, focusing instead on a custom solution suitable for the simplicity of ARC grids.

The size of the vocabulary is much smaller than in language modeling, which is a fact we can exploit. I create custom tokens that represent the 10 unique colors of each pixel, the start and end of a row, the start and end of a grid, the start and end of a pair, and the start and end of a sequence.

#### Decoder-Only Architecture:
Initially, I had planned to use an encoder-decoder architecture as the ARC Challenge has similarities to translation problems where the inputs grids in one "language" and the output grids were in another. The model would learn to translate from input to output. 

However, I have transitioned to a **decoder-only** architecture. This was done to simplify the model architecture, reduce the number of parameters and computational overhead, and unify processing. With a decoder-only transformer, I can also more easily use FlexAttention to add inductive biases to the model to reflect the relationship between pixels -> rows -> grids -> pairs -> task as well as add causal masking for the output grid of the test pair.  

#### Custom Attention Mask:

In language modeling, bidirectional, self, and causal attention are typically used. However, in the ARC Challenge, we have a very particular structure that we can exploit. For instance, not every pixel needs to attend to every other pixel. We can allow each pixel to attend and be attended to a token representing the state of a row. Similarly, we can allow the row tokens to attend and be attended to by tokens representing the state of that grid. Grid tokens can attend to and be attended to by tokens representing the state of that pair. Pair tokens can attend to and be attended by tokens representing the state of the sequence. In essence, a type of hierarchical attention. 

<div align="center">
        <img src="media/Attention%20Mask.png" alt="Attention Mask" width="400"/>
        <p><em>Figure 1: Visualization of the custom attention mask for ARC task. The y-axis represents the index of the query token, and the x-axis represents the index of the key token. Blank cells are where attention is not allowed. The colored cells indicate where attention is allowed.</em></p>
</div>

Further, we can allow the input and output grids for training as well as the input grid for testing to attend to each other. In contrast, for the output grid of the test pair, we only want causal attention since we don't want the model to cheat on the task by looking ahead.

<div align="center">
        <img src="media/Attention%20Mask%20Breakdown.png" alt="Attention Mask" width="400"/>
        <p><em>Figure 2: Visualization showing a breakdown of the custom attention mask. The hierarchy of the query indices are broken down from most general to most specific: Sequence -> Pair -> Grid -> Row -> Pixel.</em></p>
</div>

<div align="center">
        <img src="media/Attention%20Mask%20Areas.png" alt="Attention Mask" width="400"/>
        <p><em>Figure 3: Visualization showing the main areas of the custom causal attention mask. The prefix area only applies hierarchical attention, while the test area which is comprised of the test's output grid applies hierarchical and causal attention.</em></p>
</div>

Given this custom attention masking, we hope the model will learn more efficiently due to the inductive biases encoded in the hierarchy and also with a reduced memory footprint given the sparsity of the attention mask.

#### Positional Encoding:

Given that within grids the relative position of values matters as it does in language modeling, I will use positional encodings. Here we add positional encodings to all tokens. 

#### Adaptation to New Tasks:
Unlike in general translation problems, we don't have abundant data to learn the mappings between languages (input-output pairs). Rather, we must learn to translate to new "languages" (tasks) during inference. 

To enable the model to adapt to new tasks, I will add additional layers with few trainable parameters that can be fine-tuned during test time.

By employing a technique known as grokking, which involves training for a relatively high number of epochs, as well as overparametrization, which involves using a large number of parameters relative to the training data, the model may exhibit a double-descent behavior, smoothing out learned representations to generalize effectively to new inputs.

#### Training Stages:

##### Pretraining:

I plan to use both _generative_ and _contrastive_ self-supervised training methods for pretraining. 

_Generative_ Pretraining involves recreating a patched portion of the original data. By patching the input or outputs of identical grids, the network will be incentivised to learn internal representations to understand the data. This could be done early in the training process for the network to get a basic grasp of the data. Then, by patching the input or outputs of non-identical pairs, the network will be incentivised to learn how to map from input to outputs logically. To enable faster learning, a progressive patching strategy could be used such that patches are small initially and progressively get larger as the network becomes more capable. 

_Contrastive_ Pretraining involves learning from unlabeled data by leveraging the fact that certain data is inherently more similar to other data and vice versa. Concretely, this could be done in two distinct ways:
1. The first is by leveraging the fact that the input and output of a pair should be more similar than to those of other pairs.
2. The second way is by leveraging the fact that the _mapping_ from input to output across all pairs in the same task should be more similar than to the _mappings_ of input to outputs of other tasks. This second contrastive approach might be more challenging to implement, but might enable to network to learn about mappings from input to output. 

##### Supervised Training:

Using the labeled training data, we could employ supervised training. This could also be applied during test time on the few examples provided at test time. As mentioned earlier, grokking and overparametrization may help, as well as data augmentation and regularization techniques for better generalization. 

##### Alignment Training:

Once pre-training is complete and supervised training data is exhausted, I might employ alignment techniques such as training a reward model which could rate solutions or perhaps use an actor-critic approach.  

## This repository 

This repository will document my journey and progress towards developing a solution for the ARC Challenge. Stay tuned for updates as I implement and refine my approach.

#### Getting Started:

- To run pretraining run `python pretrain.py`.
- To view tensorboard run `tensorboard --logdir ./logs/pretrain`.

