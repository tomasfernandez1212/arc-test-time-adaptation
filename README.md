# ARC Challenge and My Approach
Introduction to the ARC Challenge
Artificial General Intelligence (AGI) research has reached an impasse, with current AI models struggling to adapt to new problems or create novel solutions without extensive training. The ARC (Abstraction and Reasoning Corpus) benchmark, introduced by François Chollet in 2019, aims to address this by measuring a system’s ability to acquire new skills and solve open-ended problems. Unlike traditional AI benchmarks that focus on specific skills, ARC evaluates general intelligence, making it a challenging yet crucial test for advancing AGI.

## The ARC Prize
To incentivize progress, the ARC Prize offers over $1,000,000 to anyone who can develop and open-source a solution that beats the ARC-AGI benchmark. This competition, hosted by Mike Knoop (Co-founder of Zapier) and François Chollet (Creator of ARC-AGI and Keras), invites innovators to break the current limitations of AI and contribute to the open-source community.

## My Approach
### Goal
My goal is to create a transformer encoder-decoder capable of natively perceiving and generating data in the domain of integer grids, the format used in the ARC tasks.

### Strategy
#### Custom Tokenization and Embedding:

Existing transformers struggle with ARC tasks due to inadequate grid tokenization. Current tokenizers break grids into arbitrary characters, which is inefficient.

<div align="center">
        <img src="media/Example%20issues%20with%20existing%20tokenizers.png" alt="Example issues with existing tokenizers" width="400"/>
</div>

I will develop a simpler tokenizer and embedding space tailored for the integer grids. This approach will avoid the complexity of Vision Transformers' patching and tokenization, focusing instead on a custom solution suitable for the simplicity of ARC grids.

#### Encoder-Decoder:
Unlike modern decoder-only LLMs, I will use a encoder-decoder architecture similar to those used translate problems. 

For example, given a new task with 2 "pairs" of "input" and "output" for training and 1 unpaired input for testing, I will pass all 3 _inputs_ to the encoder and the 2 _outputs_ to the decoder. The decoder should then proceed to generate the output for the test. 

Learning a mapping from input to output grids is similar to translation, hence the use of encoder-decoder architecture. 

#### Positional Encoding:

Given that within grids the relative position of values matters as it does in language modeling, I will use positional encodings. There needs to be encoding of positions within a grid as well as encoding of positions between grids. The latter is such that the model can understand that input grid i corresponds to output grid i. 

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

