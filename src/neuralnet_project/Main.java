/**
 * 
 */
package neuralnet_project;

import java.io.IOException;
import java.util.Arrays;

import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.core.events.LearningEvent;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.Perceptron;
import org.neuroph.nnet.learning.BackPropagation;
import org.neuroph.nnet.learning.LMS;
import org.neuroph.samples.convolution.MNISTDataSet;
import org.neuroph.util.TransferFunctionType;

/**
 * @author bobboau
 *
 */
public class Main {
	
	static private String DATA_DIRECTORY = "data/";

	/**
	 * @param args
	 */
	public static void main(String[] args) {
        try {
        	
        	//make the training and test data sets
        	DataSet training_set = MNISTDataSet.createFromFile(
        		DATA_DIRECTORY+MNISTDataSet.TRAIN_LABEL_NAME,
        		DATA_DIRECTORY+MNISTDataSet.TRAIN_IMAGE_NAME,
        		200
        	);
        	DataSet test_set = MNISTDataSet.createFromFile(
        		DATA_DIRECTORY+MNISTDataSet.TEST_LABEL_NAME,
        		DATA_DIRECTORY+MNISTDataSet.TEST_IMAGE_NAME,
        		10000
        	);
        	
        	//get a trained neaural net
        	NeuralNetwork<BackPropagation> neural_net = train(training_set);
        	
    		System.out.println("Testing network.");
        	
        	//see how well it does with the test set
    		float success = evaluate(neural_net, test_set);
    		
    		System.out.println("Neural net evaluated test set with "+success*100+"% accuracy.");
        } catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	private static NeuralNetwork<BackPropagation> train(DataSet training_set){
		System.out.println("Making network.");
        NeuralNetwork<BackPropagation> neural_network = new MultiLayerPerceptron(TransferFunctionType.SIGMOID, training_set.getInputSize(), 60, 30, training_set.getOutputSize());
        ExtBackPropigation backPropagation = new ExtBackPropigation();
        backPropagation.setMaxIterations(500);
        backPropagation.setLearningRate(0.1);
        backPropagation.setMaxError(0.001);
        backPropagation.setBatchSize(50);
		System.out.println("Training network.");
        neural_network.learn(training_set, backPropagation);
        return neural_network;
    }
	
	private static float evaluate(NeuralNetwork<BackPropagation> neural_net, DataSet test_set){
		int number_right = 0;
		int cur = 0;
		for(DataSetRow data_row : test_set.getRows()) {
			neural_net.setInput(data_row.getInput());
			neural_net.calculate();
			
			double[] actual_output = neural_net.getOutput();
			
			//figure out which class is the most likely
			double greatest_probability = -1;
			int greatest_idx = -1;
			for(int i = 0; i<actual_output.length; i++){
				if(greatest_probability < actual_output[i]){
					greatest_probability = actual_output[i];
					greatest_idx = i;
				}
			}
		
			//check against desired
			double[] desired_output = data_row.getDesiredOutput();
			boolean is_correct = true;
			for(int i = 0; i<desired_output.length; i++){
				if((i == greatest_idx) != (desired_output[i] == 1.0)){
					is_correct = false;
				}
			}
			if(is_correct){
				number_right++;
			}
			
			System.out.println("     Evaluating... "+(((float)++cur)/((float)test_set.getRows().size())*100)+"%, so far it's looking like "+(((float)number_right)/((float)cur)*100)+"%");
		}
		return ((float)number_right)/((float)test_set.getRows().size());
	}
}
