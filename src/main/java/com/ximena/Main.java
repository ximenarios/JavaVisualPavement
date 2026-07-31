package com.ximena;

import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

import java.io.File;

/**
 * Main entry point for the JavaVisualPavement application.
 * This class coordinates the data loading, model building, training, 
 * evaluating, and saving processes.
 */
public class Main {
    
    // Constants for network configuration and training
    private static final int IMAGE_HEIGHT = 224;
    private static final int IMAGE_WIDTH = 224;
    private static final int CHANNELS = 3;
    private static final int NUM_CLASSES = 15;
    private static final int BATCH_SIZE = 32;
    private static final int EPOCHS = 1;

    public static void main(String[] args) {
        System.out.println("Starting JavaVisualPavement...");
        
        // Define the specific paths
        String trainDataPath = "dataset/training";
        String testDataPath = "dataset/testing";
        String modelSavePath = "dataset/pavement_model.zip";
        
        try {
            // 1. Initialize the dataset loader pointing to the training folder
            System.out.println("1. Loading training dataset from: " + trainDataPath);
            ImageDatasetLoader loader = new ImageDatasetLoader();
            ImageRecordReader trainRecordReader = loader.getRecordReader(trainDataPath);
            
            // 2. Build the CNN architecture
            System.out.println("2. Building the Convolutional Neural Network...");
            MultiLayerConfiguration conf = VisualPavementModel.buildModel(
                IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS, NUM_CLASSES
            );
            MultiLayerNetwork network = new MultiLayerNetwork(conf);
            network.init();
            
            // 3. Set up the training data iterator
            DataSetIterator trainIterator = new RecordReaderDataSetIterator(
                trainRecordReader, BATCH_SIZE, 1, NUM_CLASSES
            );
            
            // 4. Normalize the training images (scale pixel values from 0-255 to 0-1)
            System.out.println("3. Applying image normalization...");
            DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
            scaler.fit(trainIterator);
            trainIterator.setPreProcessor(scaler);
            
            // 5. Train the network
            System.out.println("4. Starting the training process for " + EPOCHS + " epochs...");
            for (int i = 0; i < EPOCHS; i++) {
                network.fit(trainIterator);
                System.out.println("   -> Epoch " + (i + 1) + " completed successfully.");
                trainIterator.reset();
            }
            
            // 6. Save the trained model to disk
            System.out.println("5. Saving the trained model to disk...");
            File locationToSave = new File(modelSavePath);
            boolean saveUpdater = true; // Saves the optimizer state to resume training later if needed
            ModelSerializer.writeModel(network, locationToSave, saveUpdater);
            System.out.println("   -> Model saved at: " + modelSavePath);
            
            // 7. Evaluate the model with unseen data (Testing dataset)
            System.out.println("6. Evaluating model performance on testing data...");
            ImageRecordReader testRecordReader = loader.getRecordReader(testDataPath);
            DataSetIterator testIterator = new RecordReaderDataSetIterator(
                testRecordReader, BATCH_SIZE, 1, NUM_CLASSES
            );
            
            // Crucial step: Apply the exact same normalization to the test data
            testIterator.setPreProcessor(scaler);
            
            Evaluation eval = network.evaluate(testIterator);
            System.out.println(eval.stats());
            
            System.out.println("Process finished successfully!");
            
        } catch (Exception e) {
            System.err.println("An error occurred: " + e.getMessage());
            e.printStackTrace();
        }
    }
}