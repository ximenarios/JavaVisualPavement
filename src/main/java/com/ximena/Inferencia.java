package com.ximena;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

import java.io.File;
import java.util.Arrays;
import java.util.List;

/**
 * This class is dedicated exclusively to inference (prediction).
 * It loads the pre-trained model and analyzes a single pavement image.
 */
public class Inferencia {

    public static void main(String[] args) {
        
        // 1. Define the specific file paths
        String modelPath = "dataset/pavement_model.zip";
        
        // Place the specific path of the test image you want to analyze here
        String imagePath = "dataset/testing/FPC/imagen_prueba.jpg"; 

        // The 15 pavement distress types in strict alphabetical order (as read by the label generator)
        List<String> fallas = Arrays.asList(
            "B", "D", "DB", "ECB", "EX", "FB", "FLF", "FLJ", 
            "FPC", "FT", "FTJ", "O", "PA", "PL", "PU"
        );

        try {
            System.out.println("Loading the neural network model...");
            
            // 2. Restore the pre-trained model from the saved zip file
            File modelFile = new File(modelPath);
            MultiLayerNetwork network = ModelSerializer.restoreMultiLayerNetwork(modelFile);
            
            System.out.println("Loading and processing the pavement image...");
            
            // 3. Load the image and resize it to the dimensions required by the CNN (224x224x3)
            File imageFile = new File(imagePath);
            NativeImageLoader loader = new NativeImageLoader(224, 224, 3);
            INDArray image = loader.asMatrix(imageFile);
            
            // 4. Apply the same normalization used during training (scale pixel values from 0-255 to 0-1)
            DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
            scaler.transform(image);
            
            // 5. Perform the prediction
            INDArray output = network.output(image);
            
            // 6. Interpret the output matrix to find the category with the highest probability
            int indiceGanador = output.argMax(1).getInt(0);
            String fallaDetectada = fallas.get(indiceGanador);
            double probabilidad = output.getDouble(indiceGanador) * 100;
            
            // 7. Print the final diagnostic results
            System.out.println("\n================ DIAGNOSTIC RESULT ================");
            System.out.println("Detected distress type: " + fallaDetectada);
            System.out.printf("AI Confidence level: %.2f%%\n", probabilidad);
            System.out.println("===================================================\n");

        } catch (Exception e) {
            System.err.println("An error occurred during inference: " + e.getMessage());
            e.printStackTrace();
        }
    }
}