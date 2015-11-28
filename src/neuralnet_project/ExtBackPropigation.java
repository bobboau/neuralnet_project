/**
 * 
 */
package neuralnet_project;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.Random;

import org.neuroph.core.Layer;
import org.neuroph.core.Neuron;
import org.neuroph.core.Weight;
import org.neuroph.core.Connection;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.nnet.learning.BackPropagation;

/**
 * @author bobboau
 *
 */
public class ExtBackPropigation extends BackPropagation {
	
	int batch_size = 10;
	public void setBatchSize(int _batch_size){
		batch_size = _batch_size;
	}

	/**
	 * 
	 */
	public ExtBackPropigation() {
		super();
		this.setBatchMode(true);
	}
	
	double last_error = 1.0;
	
	@Override
	protected void afterEpoch(){
		super.afterEpoch();
		if(last_error < this.previousEpochError){
			this.learningRate *= 0.95;
			System.out.println("     learning rate set to: "+this.learningRate+" on epoch "+this.currentIteration+" with error: "+this.previousEpochError);
		}
		else{
			System.out.println("     Error decreaseing and not changed from last on epoch "+this.currentIteration+" with error: "+this.previousEpochError);
		}
		last_error = this.previousEpochError;
    }

    /**
     * This method implements basic logic for one learning epoch for the
     * supervised learning algorithms. Epoch is the one pass through the
     * training set. This method  iterates through the training set
     * and trains network for each element. It also sets flag if conditions
     * to stop learning has been reached: network error below some allowed
     * value, or maximum iteration count
     *
     * @param trainingSet training set for training network
     */
    @Override
    public void doLearningEpoch(DataSet trainingSet) {

        // feed network with all elements from training set
        Iterator<DataSetRow> iterator = trainingSet.iterator();
        ArrayList<DataSetRow> random_ordered_set = new ArrayList<DataSetRow>();
        while (iterator.hasNext() && !isStopped()) {
        	random_ordered_set.add(iterator.next());
        }
        long seed = System.nanoTime();
        Collections.shuffle(random_ordered_set, new Random(seed));

        int count = 0;
        for( int i = 0; i<random_ordered_set.size(); i++){
            DataSetRow dataSetRow = random_ordered_set.get(i);
            // learn current input/output pattern defined by SupervisedTrainingElement
            this.learnPattern(dataSetRow);
            if(++count >= batch_size){
            	/*
            	double sum = 0;
            	int w_count = 0;
    	        for (int i = 0; i<neuralNetwork.getLayersCount(); i++) {
    	            // iterate neurons at each layer
    	            for (Neuron neuron : neuralNetwork.getLayers()[i].getNeurons()) {
    	                // iterate connections/weights for each neuron
    	                for (Connection connection : neuron.getInputConnections()) {
    	                    // for each connection weight apply accumulated weight change
    	                    Weight weight = connection.getWeight();
    	                    sum += Math.abs(weight.weightChange);
    	                    w_count++;
    	                }
    	            }
    	        }
            	System.out.println("doing batch update, avg delta is: "+(sum/w_count));
    	        */
    	        count = 0;
            	doBatchWeightsUpdate();
            }
        }
        this.totalNetworkError = getErrorFunction().getTotalError();
    }
    
}
