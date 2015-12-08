package neuralnet_project;

import java.io.BufferedReader;
import java.io.IOException;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.BackPropagation;
import org.neuroph.samples.convolution.MNISTDataSet;
import org.neuroph.util.TransferFunctionType;

import java.io.InputStreamReader;

/**
 * @author bobboau
 *
 */
public class Main {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		int BatchSize=Integer.valueOf(args[0]);
		int MaxIterations=Integer.valueOf(args[1]);
		double MaxError=Double.valueOf(args[2]);
		double LearningRate=Double.valueOf(args[3]);
		double BatchSizeDecayRate=Double.valueOf(args[4]);
		double BatchSizeRegenRate=Double.valueOf(args[5]);
		double LearningDecayRate=Double.valueOf(args[6]);
		double LearningRegenRate=Double.valueOf(args[7]);
		int DataSet = Integer.valueOf(args[8]);

		try {

			//make the training and test data sets
			DataSet training_set = MNISTDataSet.createFromFile(
					MNISTDataSet.TRAIN_LABEL_NAME,
					MNISTDataSet.TRAIN_IMAGE_NAME,
					50000
			);
			DataSet test_set = MNISTDataSet.createFromFile(
					MNISTDataSet.TEST_LABEL_NAME,
					MNISTDataSet.TEST_IMAGE_NAME,
					10000
			);
			DataSet validation_set = mnistdataread.createFromFile(
					MNISTDataSet.TRAIN_LABEL_NAME,
					MNISTDataSet.TRAIN_IMAGE_NAME,
					10000
			);
			
			DataSet[] data_sets = new DataSet[3];
			data_sets[0] = training_set;
			data_sets[1] = test_set;
			data_sets[2] = validation_set;
			
			NeuralNetwork<BackPropagation> actualnet = train(data_sets[DataSet],LearningRate,BatchSize,MaxIterations,MaxError,BatchSizeDecayRate,BatchSizeRegenRate,LearningDecayRate,LearningRegenRate);
			System.out.println("Testing network.");

			//see how well it does with the test set
			float success = evaluate(actualnet, test_set);

			System.out.println("Neural net evaluated test set with "+success*100+"% accuracy.");
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	private static NeuralNetwork<BackPropagation> train(DataSet training_set, double learning_rate,int BatchSize,int MaxIterations,double MaxError,double BatchSizeDecayRate,double BatchSizeRegenRate,double LearningDecayRate,double LearningRegenRate){
		System.out.println("Making network.");
		NeuralNetwork<BackPropagation> neural_network = new MultiLayerPerceptron(TransferFunctionType.SIGMOID, training_set.getInputSize(), 30, training_set.getOutputSize());
		ExtBackPropigation backPropagation = new ExtBackPropigation();
		backPropagation.setMaxIterations(MaxIterations);
		backPropagation.setLearningRate(learning_rate);
		backPropagation.setMaxError(MaxError);
		backPropagation.setBatchSize(BatchSize);
		backPropagation.setBatchSizeDecayRate(BatchSizeDecayRate);
		backPropagation.setBatchSizeRegenRate(BatchSizeRegenRate);
		backPropagation.setLearningDecayRate(LearningDecayRate);
		backPropagation.setLearningRegenRate(LearningRegenRate);
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