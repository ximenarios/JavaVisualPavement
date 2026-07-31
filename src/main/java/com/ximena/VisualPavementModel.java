package com.ximena;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer.PoolingType;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * This class defines the architecture of the Convolutional Neural Network (CNN)
 * used to classify pavement distress. It replicates the structure of the
 * original Python model using Deeplearning4j.
 */
public class VisualPavementModel {

    /**
     * Builds and configures the multi-layer neural network.
     *
     * @param height   The height of the input images in pixels.
     * @param width    The width of the input images in pixels.
     * @param channels The number of color channels (e.g., 3 for RGB).
     * @param classes  The total number of output categories (15 distress types).
     * @return A configured MultiLayerConfiguration ready for training.
     */
    public static MultiLayerConfiguration buildModel(int height, int width, int channels, int classes) {
        // We set a standard learning rate for the RMSProp optimizer
        double learningRate = 0.001;

        return new NeuralNetConfiguration.Builder()
            .seed(123)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .updater(new RmsProp(learningRate))
            .list()
            
            // Block 1: First convolutional layer and max pooling
            .layer(new ConvolutionLayer.Builder(3, 3)
                .nIn(channels)
                .nOut(32)
                .activation(Activation.RELU)
                .build())
            .layer(new SubsamplingLayer.Builder(PoolingType.MAX)
                .kernelSize(2, 2)
                .stride(2, 2)
                .build())
                
            // Block 2
            .layer(new ConvolutionLayer.Builder(3, 3)
                .nOut(64)
                .activation(Activation.RELU)
                .build())
            .layer(new SubsamplingLayer.Builder(PoolingType.MAX)
                .kernelSize(2, 2)
                .stride(2, 2)
                .build())
                
            // Block 3
            .layer(new ConvolutionLayer.Builder(3, 3)
                .nOut(128)
                .activation(Activation.RELU)
                .build())
            .layer(new SubsamplingLayer.Builder(PoolingType.MAX)
                .kernelSize(2, 2)
                .stride(2, 2)
                .build())
                
            // Block 4
            .layer(new ConvolutionLayer.Builder(3, 3)
                .nOut(128)
                .activation(Activation.RELU)
                .build())
            .layer(new SubsamplingLayer.Builder(PoolingType.MAX)
                .kernelSize(2, 2)
                .stride(2, 2)
                .build())
                
            // Dense Layer (Flattening is handled automatically by setInputType below)
            .layer(new DenseLayer.Builder()
                .nOut(512)
                .activation(Activation.RELU)
                .build())
                
            // Output Layer (Classifier)
            .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                .nOut(classes)
                .activation(Activation.SOFTMAX)
                .build())
                
            // Define the input type to automatically calculate tensor dimensions
            .setInputType(InputType.convolutionalFlat(height, width, channels))
            .build();
    }
}