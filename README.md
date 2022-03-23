# parallel-cnn
15-418/618 Final Project: Parallel convolutional neural networks

# Project Proposal
## Summary
We will implement convolutional neural networks (CNNs) in CUDA on the GHC and PSC GPUs, parallelizing the computation that occurs within the neural network layers. We will use data parallelism for convolutional layers and model parallelism for fully-connected layers. If we have extra time, we will use OpenMPI to increase scalability and allow us to run larger datasets across multiple processors.

## Background
The field of computer vision uses a technique called “convolutions” that allow pixels in an image to see their neighboring pixels. As an example, a vertical line detector might take the x-gradient of an image (right column - left column, a Sobel filter/convolution). Convolutional neural networks (CNNs) combine this ability with deep neural networks.

Since training CNNs requires a large amount of computation upon a large amount of data, parallelization is required to ensure that any neural network can be trained in a reasonable amount of time. Increasing training efficiency can dramatically change what complexity of tasks or datasets can be tackled.

Table 1 shows the key properties of CNN layers that affect the selection of optimal parallelization strategies for a specific neural network layer. 

|                        | % computation | % parameters | Representation size |
| ---------------------- | ------------- | ------------ | ------------------- |
| Convolutional layers   | 90-95%        | 5%           | Large               |
| Fully-connected layers | 5-10%         | 95%          | Small               |

*Table 1: Layer types with different properties (Krizhevsky 2014)*

![img](https://lh4.googleusercontent.com/jug5B05y0eKeuQ21qK4DeR83cxjlqWINRQwOUIYwwiuTpVcTtoWmC0E8RDoIOgVCUPYvtQbeXIDqRe6hUQbyL6gKE4Gz3qWsiuu37HlZ5syihv-rwKDZvC5uGbhqKkGS5Tgs9W8g)

(Krizhevsky 2014)

### Different Types of Layers need Different Parallelism
Data parallelism is a very natural parallelization strategy for convolutional layers because it can parallelize the significant computation over a large dimension (data), and it does not require significant synchronization due to the limited number of parameters. In contrast, the communication cost of the large number of parameters needed by a fully-connected layer outweighs the potential advantages that can be gained from data parallelism, and Krizhevsky proposes that model parallelism be leveraged instead. For model parallelism, the different workers will train different parts of the model, which requires appropriate synchronization between workers.

### How to switch between Model & Data Parallelism
In the data parallelism model, each worker has a different batch of images, but in the model parallelism mode, each worker gets a section of the model. Thus, workers need to communicate with each other to share batches. Krizhevsky proposes three categories of parallelization strategies for this communication and computation:

1. Communication happens at once, with all workers sharing all images, and then each worker continuing on to apply their part of the fully connected layer to each copy of the big batch of images.

    a. For memory-limited GPUs, this can pose a problem. Latency is also a factor.

2. One worker shares its (information about its) batch of images. Workers apply this batch to the next layer while another worker sends their images.

    a. A lot of latency can be hidden via this strategy, although the ratio between communication and computation depends on the number of workers

3. Workers all send a subset of their images (size of batch / number of workers), and then each worker applies their combo batch (of size equal to the normal batch size) to the next layer.

    a. This allows the amount of computation vs. communication to stay constant with different numbers of workers

We note the above specification is for the forward pass. For the backward pass, while communication is happening each worker computes the next batch’s forward pass.

## Challenges
Our main challenge will be implementing Krizhevsky’s proposed parallelization strategies for fully-connected layers. The three communication strategies each have separate challenges with different communication bottlenecks. We expect it will be difficult to coordinate our workers to hide latency by performing communication and computation tasks at the same time. Additionally, we expect challenges related to correctness because of how workers share data and switch between parallelization strategies.

Another significant challenge will be determining which parallelization strategy is best for layers that are neither fully connected nor convolutional. For example, we will need to determine whether max pooling, ReLu, and softmax layers would be better suited to data parallelism or model parallelism. We will observe the amount of computation versus the communication cost needed to synchronize parameters, and we may need to implement multiple parallelization strategies to determine the best one. We also expect that different parallelism strategies might be better based on various parameters such as the size of the convolutional kernel or specifics of different kinds of layers.

## Resources
We will use the GPUs on both the GHC and PSC machines. We will train our CNNs using some of the datasets found here: 
- https://imerit.net/blog/top-13-machine-learning-image-classification-datasets-all-pbm/
We plan to implement CNNs from scratch, largely following the parallelization strategies discussed in this article: [One weird trick for parallelizing convolutional neural networks](https://arxiv.org/pdf/1404.5997.pdfï¼‰ã€‚) (primary resource with layer-level parallelism). (Additional resources we may use are listed [here](#references).)

## Goals and deliverables
1. CUDA implementation of CNN with VGG16 architecture (with data-parallel convolution layers and sequential fully-connected layers)
   a. We will implement four layers: maxPool, ReLu, Conv2D, and softMax
   b. We’ll be using the image classification task
2. Add model parallelism to CUDA implementation of CNN
3. Performance measurements on different machines with different datasets and CNN architectures
    a. Determine whether computation or memory bandwidth is the limiting factor and profile our code to find bottlenecks. 
    b. Compare our implementation with Pytorch and TensorFlow as well as performance differences between different machines.
    c. Compare our speedup values with those found in Kriezhevsky’s paper.
4. Stretch goal: Implement CNNs with more kinds of layers (e.g. sigmoid) and dropout parameters
5. Stretch goal: CUDA implementation with OpenMPI to run larger datasets across multiple machines

We plan to achieve goals 1, 2, and 3 (as seen above). If we encounter difficulties, we will still complete 1 and 3, but we may only partially complete 2 (e.g. not fully implementing all strategies for model parallelism). If we have extra time, we may perform additional performance measurements (for 3) with different datasets, CNN architectures, and larger convolutional kernels, determining how different architectures affect speedup and performance. Also, if we have extra time, we are interested in doing 4 and 5.

## Platform Choice
We will be using C++ and CUDA as performance-oriented languages with support for data parallelism. We will be using the GPUs on the GHC and PSC machines because GPUs are the primary hardware acceleration used by machine learning. 

## Schedule
[3/20-3/26]: Research and write proposal (due 4/23)
[3/27-4/2]: Implement data-parallel conv2D and maxPool
[4/3-4/9]: Implement ReLu and softMax and overall VGG16 architecture
[4/10-4/16]: Implement fully connected layers with model parallelism (hopefully partially finish model parallelism by checkpoint 4/11)
[4/17-4/23]: Finish up model parallelism and implement a CNN with fully-connected layers (e.g. AlexNet), and/or perform performance measurements 
[4/24-4/29]: Finish up performance measurements and write final report including substantial performance analysis
[5/1-5/5]: Prepare and finish final presentation

We may do the stretch goals in the last three weeks if we have extra time.

## References
- [One weird trick for parallelizing convolutional neural networks](https://arxiv.org/pdf/1404.5997.pdfï¼‰ã€‚) (primary resource with layer-level parallelism)
- [Exploring Hidden Dimensions in Parallelizing Convolutional Neural Networks](http://proceedings.mlr.press/v80/jia18a/jia18a.pdf?ref=https://codemonkey.link) (layer-wise parallelism)
- [Evaluation of MPI_Allreduce for Distributed Training of Convolutional Neural Networks](https://ieeexplore.ieee.org/abstract/document/9407073)(MPI)
- [Step by step VGG16 implementation in Keras for beginners](https://towardsdatascience.com/step-by-step-vgg16-implementation-in-keras-for-beginners-a833c686ae6c#:~:text=VGG16 is a convolution neural,vision model architecture till date) (VGG16 model architecture)
- [AlexNet](https://en.wikipedia.org/wiki/AlexNet) (AlexNet model architecture)
- [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)
- [A CUDA-based implementation of convolutional neural network | IEEE Conference Publication](https://ieeexplore.ieee.org/document/8320682)
