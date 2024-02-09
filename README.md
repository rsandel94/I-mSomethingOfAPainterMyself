# I-mSomethingOfAPainterMyself
    CycleGAN generator of monet style paintings
    The results can be found on the link below
    https://www.kaggle.com/code/robertsandel/i-m-something-of-an-artist-myself
## Sources
    This project is done with the help of these URLS
    https://www.kaggle.com/code/hardik201003/cyclegan-i-m-something-of-a-painter-myself
    https://www.kaggle.com/code/unfriendlyai/two-objective-discriminator
    https://www.kaggle.com/code/amyjang/monet-cyclegan-tutorial/notebook


## Objective
    The primary objective of the project is to demonstrate the ability of CycleGANs to perform image-to-image translation tasks without paired training data. By leveraging adversarial training and cycle consistency, the model can learn meaningful mappings between different visual domains, enabling tasks such as style transfer, object transfiguration, and more.

    Overall, the CycleGAN project showcases the power of deep learning techniques in addressing real-world problems related to image manipulation, artistic style transfer, and domain adaptation, opening up new possibilities for creative expression and data-driven transformations in visual media.

## Key Components

    Data Preparation: The project starts with data preparation, including loading and preprocessing images from two different domains. In this case, the domains could represent Monet-style paintings and photographs.

    Model Architecture: The core of the project involves implementing the CycleGAN model architecture, which consists of two generator networks (for each domain) and two discriminator networks. The generators aim to translate images from one domain to the other, while the discriminators aim to distinguish between real and translated images.

    Adversarial Training: The CycleGAN model undergoes adversarial training, where the generators and discriminators are trained simultaneously in a minimax game. The generators aim to fool the discriminators by generating realistic images, while the discriminators aim to differentiate between real and generated images accurately.

    Cycle Consistency Loss: A key concept in CycleGANs is the cycle consistency loss, which ensures that the translation process is consistent in both directions. This loss encourages the reconstructed images (after translation and back-translation) to be similar to the original images, thus preserving important characteristics across domains.

    Evaluation and Testing: Once trained, the CycleGAN model is evaluated using various metrics to assess the quality of the translated images. Additionally, qualitative testing is performed to visually inspect the results and determine the effectiveness of the model.

### Initial Setup

    File Paths:
        The file paths for the Monet and photo datasets are specified using the sourcePath variable.
        TFRecord files for both datasets are retrieved using the tf.io.gfile.glob function, which returns a list of file paths.

    Data Loading:
        The number of Monet and photo TFRecord files is printed to confirm the availability of the data.

    Image Size Specification:
        The variable IMAGE_SIZE is set to [256, 256], indicating the desired dimensions for the images.

    Image Decoding:
        The decode_img function is defined to decode the image from JPEG format.
        Pixel values are normalized to the range [-1, 1] and reshaped to the specified image size.

    TFRecord Reading:
        The read_tfrec function is defined to read a single TFRecord example.
        It parses the example according to the specified format and decodes the image.

    Data Loading (Continued):
        The load_data function loads TFRecord datasets from the provided files and applies the read_tfrec function to each example.
        Both Monet and photo datasets are loaded and returned as TFRecordDataset objects.

    Batching Data:
        The loaded datasets are batched with a batch size of 1, indicating that each batch contains a single image.

    Example Retrieval:
        Example images from both the Monet and photo datasets are retrieved using the next(iter(dataset)) method, which returns the next element of the dataset iterator.


### Building the Generators

    The generator network is responsible for transforming input images from one domain (e.g., photographs) to another domain (e.g., Monet-style paintings). It consists of several downsampling layers followed by upsampling layers.

    Downsampling Layers: These layers reduce the spatial dimensions of the input image while extracting high-level features. Each downsampling layer typically consists of a convolutional operation followed by an activation function (such as Leaky ReLU) to introduce non-linearity.

    Upsampling Layers: After the input image is processed through the downsampling layers, it undergoes a series of upsampling layers. These layers gradually increase the spatial dimensions of the image to reconstruct the transformed output. Each upsampling layer typically consists of a transpose convolutional operation followed by an activation function.

    Skip Connections: To preserve fine-grained details during the transformation process, skip connections are used. These connections allow information from earlier layers to bypass several layers and be directly concatenated with later layers. This helps in maintaining spatial information and enhancing the quality of the generated output.

    Final Output: The final layer of the generator network consists of a transpose convolutional operation with a 'tanh' activation function. This layer produces the transformed output image with the desired characteristics (e.g., Monet-style features).

    By stacking multiple downsampling and upsampling layers, along with skip connections, the generator network learns to effectively map input images from one domain to another, producing high-quality transformed outputs.


### Initializing the Generator and Discriminator Objects
    Generator Initialization:
        The generator() function is called to create the generator network. This network is responsible for transforming input photos into Monet-esque images.
        monet_generator is assigned as the instance of the generator network that will perform the transformation from photos to Monet-esque images.

    Discriminator Initialization:
        The discriminator() function is invoked to create the discriminator network. This network is designed to distinguish between real Monet-esque images and generated Monet-esque images.
        monet_discriminator is instantiated as the discriminator network, which will classify Monet-esque images.

    Additional Initialization:
        Another pair of generator and discriminator objects (photo_generator and photo_discriminator) are initialized. These networks are intended for the reverse transformation, generating photos from Monet-esque images and classifying real and generated photos, respectively.

    Generating Sample Output:
        To visualize the initial performance of the generator, a sample input photo (ex_photo) is passed through monet_generator. This generates a Monet-esque image (photo_to_monet) from the input photo.
        Matplotlib subplots are created to display the original input photo and the generated Monet-esque photo side by side.

    This initialization step sets up the generator and discriminator networks, allowing for the subsequent training and evaluation of the CycleGAN model. The visualization of the generated Monet-esque photo provides an initial insight into the generator's performance before actual data fitting.

### The CycleGAN function
    The CycleGAN class is defined, inheriting from the keras.Model class. It serves as the core of the CycleGAN model.
    Constructor:
        Accepts the generator and discriminator networks for both Monet-esque images (monet_gen, monet_disc) and photos (photo_gen, photo_disc), along with a parameter lambda_cycle controlling the importance of cycle consistency loss.
        Initializes the model with these networks.
    compile Method:
        Configures the model for training.
        Accepts optimizers for the generator and discriminator networks, loss functions for generator, discriminator, cycle consistency, and identity, which are then stored in the model.
    train_step Method:
        Automatically invoked during training when the fit() method is called.
        Performs a single training step.
        Computes the generator and discriminator losses, applies gradients, and updates the model parameters using the specified optimizers.

### Loss Functions
    gen_loss_fn:
        Computes the generator loss using binary cross-entropy loss for generated images.
    disc_loss_fn:
        Computes the discriminator loss using binary cross-entropy loss for both real and generated images.
    cycle_loss_fn:
        Computes the cycle consistency loss by measuring the absolute difference between real and cycled images.
        Multiplies the loss by the lambda_cycle parameter to control its weight in the overall loss.
    identity_loss_fn:
        Computes the identity loss by measuring the absolute difference between real and reconstructed images.
        Multiplies the loss by a factor of Lambda for normalization.

### Optimizers
    Adam optimizers are used for both the generator and discriminator networks (m_gen_opt, m_disc_opt, p_gen_opt, p_disc_opt).
    Learning rate is set to 2e-4 and beta_1 parameter is set to 0.5 for all optimizers.

### Compiling and training
    Creating CycleGAN Model Instance:
        An instance of the CycleGAN class is created, passing the generator and discriminator networks for both Monet-esque images (monet_generator, monet_discriminator) and photos (photo_generator, photo_discriminator).
        The lambda_cycle parameter is set to 10.

    Compiling the Model:
        The compile() method of the CycleGAN model is called.
        Optimizers (m_gen_opt, m_disc_opt, p_gen_opt, p_disc_opt) and loss functions (gen_loss_fn, disc_loss_fn, cycle_loss_fn, identity_loss_fn) are passed to configure the model for training.

    Training/Fitting the Model:
        The fit() method of the compiled CycleGAN model is called to train the model.
        Training data is provided as a zipped dataset of Monet-esque images (monet_data) and photos (photo_data).
        The training process runs for 30 epochs, iterating over the dataset for each epoch.

In summary, the CycleGAN model is compiled with specified optimizers and loss functions, then trained on paired datasets of Monet-esque images and photos for 30 epochs. This process enables the model to learn to translate images between the two domains while optimizing the specified objectives and losses.

