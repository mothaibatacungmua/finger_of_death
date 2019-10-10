#include <torch/torch.h>
#include <iostream>
#include <cstdio>
#include <cmath>
#include <string>

using namespace torch;

// The size of the noise vector fed to the generator.
const int64_t kNoiseSize = 100;

// The batch size for training.
const int64_t kBatchSize = 64;

// The number of epochs to train.
const int64_t kNumberOfEpochs = 30;

// Set to `true` to restore models and optimizers from previously saved
// checkpoints.
const bool kRestoreFromCheckpoint = false;

// After how many batches to log a new update with the loss value.
const int64_t kLogInterval = 10;


struct DCGANGeneratorImpl : nn::Module {
    DCGANGeneratorImpl (int kNoiseSize)
        : conv1(nn::Conv2dOptions(kNoiseSize, 256, 4)
                    .with_bias(false)
                    .transposed(true)),
          batch_norm1(256),
          conv2(nn::Conv2dOptions(256, 128, 3)
                    .stride(2)
                    .padding(1)
                    .with_bias(false)
                    .transposed(true)),
          batch_norm2(128),
          conv3(nn::Conv2dOptions(128, 64, 4)
                    .stride(2)
                    .padding(1)
                    .with_bias(false)
                    .transposed(true)),
          batch_norm3(64),
          conv4(nn::Conv2dOptions(64, 1, 4)
                    .stride(2)
                    .padding(1)
                    .with_bias(false)
                    .transposed(true))

    {
        register_module("conv1", conv1);
        register_module("conv2", conv2);
        register_module("conv3", conv3);
        register_module("conv4", conv4);
        register_module("batch_norm1", batch_norm1);
        register_module("batch_norm2", batch_norm2);
        register_module("batch_norm3", batch_norm3);
    }

    torch::Tensor forward(torch::Tensor x){
        x = torch::relu(batch_norm1(conv1(x)));
        x = torch::relu(batch_norm2(conv2(x)));
        x = torch::relu(batch_norm3(conv3(x)));
        x = torch::relu(conv4(x));
        return x;
    }

    nn::Conv2d conv1, conv2, conv3, conv4;
    nn::BatchNorm batch_norm1, batch_norm2, batch_norm3;
};

TORCH_MODULE(DCGANGenerator);


struct DCGANDiscriminatorImpl : nn::Module {
    DCGANDiscriminatorImpl()
        : conv1(nn::Conv2dOptions(1, 64, 4)
                    .stride(2)
                    .padding(1)
                    .with_bias(false)),
          batch_norm1(64),
          conv2(nn::Conv2dOptions(64, 128, 4)
                    .stride(2)
                    .padding(1)
                    .with_bias(false)),
          batch_norm2(128),
          conv3(nn::Conv2dOptions(128, 256, 4)
                    .stride(2)
                    .padding(1)
                    .with_bias(false)),
          batch_norm3(256),
          conv4(nn::Conv2dOptions(256, 1, 3)
                    .stride(1)
                    .padding(0)
                    .with_bias(false))
    {
        register_module("conv1", conv1);
        register_module("conv2", conv2);
        register_module("conv3", conv3);
        register_module("conv4", conv4);
        register_module("batch_norm1", batch_norm1);
        register_module("batch_norm2", batch_norm2);
        register_module("batch_norm3", batch_norm3);
    };

    torch::Tensor forward(torch::Tensor x){
        x = torch::relu(batch_norm1(conv1(x)));
        x = torch::relu(batch_norm2(conv2(x)));
        x = torch::relu(batch_norm3(conv3(x)));
        x = torch::tanh(conv4(x));
        return x;
    }

    nn::Conv2d conv1, conv2, conv3, conv4;
    nn::BatchNorm batch_norm1, batch_norm2, batch_norm3;
        
};

TORCH_MODULE(DCGANDiscriminator);



int main(int argc, char *argv[]){
    torch::manual_seed(1);
    torch::Device device(torch::kCPU);
    string dataPath("./mnist");
    if(argc == 2){
        dataPath.assign(argv[1]);
    }
    DCGANGenerator generator(kNoiseSize);
    DCGANDiscriminator discriminator;

    generator->to(device);
    discriminator->to(device);

    auto dataset = torch::data::datasets::MNIST(dataPath)
                        .map(torch::data::transforms::Normalize<>(0.5, 0.5))
                        .map(torch::data::transforms::Stack<>());

    const int64_t batches_per_epoch = 
        std::ceil(dataset.size().value()) / static_cast<double>(kBatchSize);
    auto data_loader = torch::data::make_data_loader(
        std::move(dataset),
        torch::data::DataLoaderOptions().batch_size(kBatchSize).workers(2));

    torch::optim::Adam generator_optimizer(
        generator->parameters(), torch::optim::AdamOptions(2e-4).beta1(0.5)
    );
    torch::optim::Adam discriminator_optimizer(
        discriminator->parameters(), torch::optim::AdamOptions(2e-4).beta1(0.5)
    );
    
    if(kRestoreFromCheckpoint){
        torch::load(generator, "generator-checkpoint.pt");
        torch::load(generator_optimizer, "generator-optimizer-checkpoint.pt");
        torch::load(discriminator, "discriminator-checkpoint.pt");
        torch::load(discriminator_optimizer, "discriminator-optimizer-checkpoint.pt");
    }

    for (int64_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch){
        int64_t batch_index = 0;
        for (torch::data::Example<>& batch : *data_loader){
            discriminator->zero_grad();
            // train discriminator with real images
            torch::Tensor real_images = batch.data.to(device);
            torch::Tensor real_labels = 
                torch::empty(batch.data.size(0), device).uniform_(0.8, 1.0);
            torch::Tensor real_output = discriminator->forward(real_images);
            torch::Tensor d_loss_real = 
                torch::binary_cross_entropy(real_output, real_labels);
            d_loss_real.backward();

            // train discriminator with fake images
            torch::Tensor noise = 
                torch::randn({batch.data.size(0), kNoiseSize, 1, 1}, device);
            torch::Tensor fake_images = generator->forward(noise);
            torch::Tensor fake_labels = torch::zeros(batch.data.size(0), device);
            torch::Tensor fake_output = discriminator->forward(fake_images.detach());
            torch::Tensor d_loss_fake = 
                torch::binary_cross_entropy(fake_output, fake_labels);
            d_loss_fake.backward();

            torch::Tensor d_loss = d_loss_real + d_loss_fake;
            discriminator_optimizer.step();

            // Train generator 
            generator->zero_grad();
            fake_labels.fill_(1);
            fake_output = discriminator->forward(fake_images);
            torch::Tensor g_loss =
                torch::binary_cross_entropy(fake_output, fake_labels);
            g_loss.backward();
            generator_optimizer.step();

            batch_index++;
            
            if (batch_index % kLogInterval == 0){
                std::printf(
                    "\r[%2ld/%2ld][%3ld/%3ld] D_loss: %.4f | G_loss: %.4f\n",
                    epoch,
                    kNumberOfEpochs,
                    batch_index,
                    batches_per_epoch,
                    d_loss.item<float>(),
                    g_loss.item<float>()
                );
            }

        }
    }
    return 0;
}