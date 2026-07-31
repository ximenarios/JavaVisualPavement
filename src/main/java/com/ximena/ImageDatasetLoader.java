package com.ximena;

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;

import java.io.File;
import java.io.IOException;
import java.util.Random;

/**
 * This class is responsible for loading and preprocessing the pavement images.
 * It reads the images from the local dataset directory, resizes them to a uniform scale,
 * and extracts the classification labels based on the folder names.
 */
public class ImageDatasetLoader {
    
    // Define the target image dimensions (e.g., 224x224 pixels with 3 color channels: RGB)
    private static final int HEIGHT = 224;
    private static final int WIDTH = 224;
    private static final int CHANNELS = 3;

    /**
     * Creates and initializes an ImageRecordReader to process the dataset.
     * 
     * @param datasetPath The absolute or relative path to the main dataset folder.
     * @return An initialized ImageRecordReader ready to feed the neural network.
     * @throws IOException If the directory cannot be read.
     */
    public ImageRecordReader getRecordReader(String datasetPath) throws IOException {
        File mainPath = new File(datasetPath);
        
        // FileSplit randomizes the order of the images for better training results
        FileSplit fileSplit = new FileSplit(mainPath, NativeImageLoader.ALLOWED_FORMATS, new Random(123));
        
        // Extract the label (distress category) from the parent directory name automatically
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        
        // Initialize the ImageRecordReader with the target dimensions
        ImageRecordReader recordReader = new ImageRecordReader(HEIGHT, WIDTH, CHANNELS, labelMaker);
        recordReader.initialize(fileSplit);
        
        return recordReader;
    }
}