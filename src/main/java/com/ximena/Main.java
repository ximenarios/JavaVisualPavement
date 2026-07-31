package com.ximena;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

/**
 * Main entry point for the JavaVisualPavement application.
 * This class coordinates the data loading, model building, and training processes.
 */
public class Main {
    
    // Constants for network configuration
    private static final int IMAGE_HEIGHT = 224;
    private static final int IMAGE_WIDTH = 224;
    private static final int CHANNELS = 3;
    private static final int NUM_CLASSES = 15;

    public static void main(String[] args) {
        System.out.println("Starting JavaVisualPavement...");
        System.out.println("Initializing the deep learning environment...");
        
        String datasetPath = "dataset/";
        
        try {
            // 1. Initialize the dataset loader
            ImageDatasetLoader loader = new ImageDatasetLoader();
            System.out.println("Ready to load images from: " + datasetPath);
            
            // 2. Build the CNN architecture
            System.out.println("Building the Convolutional Neural Network...");
            MultiLayerConfiguration conf = VisualPavementModel.buildModel(
                IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS, NUM_CLASSES
            );
            
            // 3. Instantiate and initialize the network
            MultiLayerNetwork network = new MultiLayerNetwork(conf);
            network.init();
            
            System.out.println("Model built successfully!");
            System.out.println(network.summary()); // Prints the architecture details
            
        } catch (Exception e) {
            System.err.println("An error occurred while initializing the application: " + e.getMessage());
            e.printStackTrace();
        }
    }
}