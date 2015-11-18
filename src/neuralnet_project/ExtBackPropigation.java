/**
 * 
 */
package neuralnet_project;

import org.neuroph.nnet.learning.BackPropagation;

/**
 * @author bobboau
 *
 */
public class ExtBackPropigation extends BackPropagation {

	/**
	 * 
	 */
	public ExtBackPropigation() {
		super();
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

}
