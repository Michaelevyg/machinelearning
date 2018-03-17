package HomeWork1;

import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

public class LinearRegression implements Classifier {
	
    private int m_ClassIndex;
	private int m_truNumAttributes;
	private double[] m_coefficients;
	private double m_alpha;
	
	//the method which runs to train the linear regression predictor, i.e.
	//finds its weights.
	@Override
	public void buildClassifier(Instances trainingData) throws Exception {
		trainingData = new Instances(trainingData);
		m_ClassIndex = trainingData.classIndex();
		//since class attribute is also an attribute we subtract 1
		m_truNumAttributes = trainingData.numAttributes() - 1;
		findAlpha(trainingData);
		m_coefficients = gradientDescent(trainingData);
	}
	
	private void findAlpha(Instances data) throws Exception {
		double bestAlpha = Math.pow(3, -17);
		double bestError = Double.MAX_VALUE;
		for (int i = -17; i <= 2; i++ ) {
			this.m_alpha = Math.pow(3, i);
			// initiate the coefficient array after every Alpha iteration
			this.m_coefficients = gradientDescent(data);
			double currentError = calculateMSE(data); 
			//update the alpha if we got better error
			if (currentError < bestError ) {
				bestAlpha = this.m_alpha;
			}	
		}
		this.m_alpha = bestAlpha;
	}
	
	/**
	 * An implementation of the gradient descent algorithm which should
	 * return the weights of a linear regression predictor which minimizes
	 * the average squared error.
     * 
	 * @param trainingData
	 * @throws Exception
	 */
	private double[] gradientDescent(Instances trainingData)
			throws Exception {
		//ignore the last value because its a class and not an attribute
		int attrSize = trainingData.numAttributes() ;
		this.m_coefficients = new double [attrSize]; 
		
		//initiate coffArray
		for (int i = 0; i < attrSize ; i++) {
			this.m_coefficients[i] = 1;
		}
		
		double currentError = 0;
		double prevError = Double.MAX_VALUE;
		//for every alpha we run 20000 iterations
		for (int j = 1; j < 20000 ; j++) {
			//iteration for the current alpha
			this.m_coefficients = coffCalc (trainingData);
		
			if ( j % 100 == 0 ) {
				currentError = calculateMSE(trainingData);
				if (Math.abs(currentError-prevError) < 0.003) {
					break;
				}
			prevError = currentError;
			}	
		}
		return coffCalc(trainingData);
	}
	
	
	/**
	 * func for calculating the coefficients array using the alpha we found
	 * @param trainingData
	 * @return the updated coefficients array
	 */
	private double[] coffCalc (Instances trainingData) {
		int attrSize = trainingData.numAttributes() ;
		double [] temp = new double [attrSize];	
		
		for (int i = 0; i < attrSize ; i++ ) {
		//update all coefficients values using gradient helper which is a function for calculating the sigma operation.
		temp[i] = (this.m_coefficients[i]) -((this.m_alpha)/(attrSize))*gradientHelper(trainingData, i, this.m_coefficients);
		}
		return temp;
	}
	
	/**
	 * helper function for calculating the sigma 
	 * @param trainingData
	 * @param tetta
	 * @param coffArray
	 * @return sigma
	 */
	private double gradientHelper (Instances trainingData , int tetta, double [] coffArray) {
		double temp = 0;
		for (int i = 0; i < trainingData.numInstances() ; i++) {
				temp += (vectorMult(trainingData.instance(i), coffArray) - trainingData.instance(i).classValue())
						*trainingData.instance(i).value(tetta);
			}
		return temp;
	}
	
	/**
	 * helper function for calculating the vector multiplication 
	 * @param instance
	 * @param coffArray
	 * @return the result of the vector multiplication
	 */
	private double vectorMult (Instance instance , double [] coffArray) {
		double vectorMul = coffArray[0];
		for (int i = 1; i < instance.numAttributes(); i++ ) {
			//the i tetta related to the i-1 attribute
			vectorMul += coffArray[i]*instance.value(i-1); 
		}
		return vectorMul;
	}
	
	/**
	 * Returns the prediction of a linear regression predictor with weights
	 * given by m_coefficients on a single instance.
     *
	 * @param instance
	 * @return
	 * @throws Exception
	 */
	public double regressionPrediction(Instance instance) throws Exception {
		return vectorMult (instance , this.m_coefficients );
	}
	
	/**
	 * Calculates the total squared error over the data on a linear regression
	 * predictor with weights given by m_coefficients.
     *
	 * @param testData
	 * @return
	 * @throws Exception
	 */
	public double calculateMSE(Instances data) throws Exception {
		double sumError = 0;
		for (int i = 0; i < data.numInstances(); i++) {
			sumError += Math.pow(regressionPrediction(data.instance(i)) - data.instance(i).classValue(), 2);
		}
		sumError /= 2*data.numInstances();
		return sumError;
	}
	
	/**
	 * this func printing the coefficients
	 */
	public void printingCoefficients (){
		for(int i = 0; i < m_coefficients.length; i++){
			System.out.print(" "+ m_coefficients[i]);
		}
	}
    
    @Override
	public double classifyInstance(Instance arg0) throws Exception {
		// Don't change
		return 0;
	}

	@Override
	public double[] distributionForInstance(Instance arg0) throws Exception {
		// Don't change
		return null;
	}

	@Override
	public Capabilities getCapabilities() {
		// Don't change
		return null;
	}
}
