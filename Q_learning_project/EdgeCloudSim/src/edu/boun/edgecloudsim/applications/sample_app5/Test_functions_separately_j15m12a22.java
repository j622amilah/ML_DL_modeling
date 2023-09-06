
package edu.boun.edgecloudsim.applications.sample_app5;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.trees.RandomForest;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.CSVLoader;

import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.unsupervised.attribute.NumericToBinary;
import weka.filters.supervised.instance.SMOTE;



import java.io.File;  // Import the File class
import java.io.IOException;


public class Test_functions_separately_j15m12a22 {
	
	// ******************* Global variables for this class only *******************
	// q-learning parameters
	private double EPSILON = 0.5;
	private double GAMMA = 0.995; //discount factor
	public static double ALPHA = 1e-3; 	//learning rate  
	private double NU = 0.1;
	
	private static final String[] STATE_DATA_ATTRIBUTES = {"Tall","Cj_rem","processing_time", "CPUcost", "BWcost", "packet_loss_rate", "getNetworkDelay", "result_of_task", "QOE", "y_actual_binary"};

	static Instances state_train;
	
	private int cnt; //count of running through the q_learning algorithm
	private int total_points;
	
	// ******************* 
	
	
	public void main(String[] args) throws Exception {
		
		// sim is running - vehicularEdgeOrchestrator starts episode, a
		episodeStarted();
		
		if (cnt == 0){
			initializing_episode();
		}
		
	}
	
	
	// want to do this when starting an episode : 
	public void episodeStarted() {
		double startTime = System.currentTimeMillis();
		total_points = 0;
		cnt = 0; // counter for periodic update of q_network weights
	}
	
	
	
	
	
	public void initializing_episode() throws Exception {
		// -----------------------------------------------
		// [Step 1] Gather input data
		System.out.println("[Step 1] Gather input data\n");
		state_train = read_data_from_file();
		// -----------------------------------------------
		
		
		// -----------------------------------------------
		// [Step 2] Train model: Defines the model as 'MultilayerPerceptron'
		System.out.println("[Step 2] Train model\n");
		MultilayerPerceptron q_network = train_qlearning_nns(state_train);
		MultilayerPerceptron target_q_network = train_qlearning_nns(state_train);
		// -----------------------------------------------
		
		
		// -----------------------------------------------
		// [Step 3] Verify that initial trained model is decent
		System.out.println("[Step 3] Evaluate q_network model\n");
		evaluate_model(q_network);
		
		System.out.println("[Step 3] Evaluate target_q_network model\n");
		evaluate_model(target_q_network);
		// -----------------------------------------------
		
		// -----------------------------------------------
		// [Step 4] Start each episode
		System.out.println("[Step 4] Start each episode\n");
		
		//double [] a_t = new double[update_interval];  // double à chaque instance
		//double [][] s_t  = new double[update_interval][9];  //vector à chaque instance
		//double [] R_t = new double[update_interval];
		//double [] y_t = new double[update_interval];
		//double [] d_t = new double[update_interval];
		// -----------------------------------------------;
	}
	
	
	
	
	public static Instances read_data_from_file() throws Exception {
		
		String CSV_FilePath = "C:\\Users\\jafouche\\Documents\\Edge_computing\\XNETmatrix.txt";
		CSVLoader loader = new CSVLoader();
		String[] options = new String[5];
		options[0] = "-H";
		options[1] = "-F";
		options[2] = ",";
		options[3] = "-R";
		options[4] = "1,2,3,4,5,6,7,8,9,10-last";
		loader.setOptions(options);
		loader.setSource(new File(CSV_FilePath));
		loader.setNoHeaderRowPresent(true);
		Instances state_train = loader.getDataSet();
		
		// Set the class
		System.out.println("Number of columns=state_train.numAttributes(): " + state_train.numAttributes());
		state_train.setClassIndex(state_train.numAttributes() - 1);
		
		// -----------------------
		// setup a filter to transform the numerical values in the last column to nominal/'string' 
		// smote can only use non-numeric class values
		NumericToNominal filter1 = new NumericToNominal(); // Converting last index attribute to type
		filter1.setAttributeIndices("last"); // as the last index would be class
		filter1.setInputFormat(state_train);  // specify the input instance format for the filter
		state_train = Filter.useFilter(state_train, filter1); // filtered
		
		// Balance the classes 
		SMOTE smote=new SMOTE();
	    smote.setInputFormat(state_train);       
	    state_train = Filter.useFilter(state_train, smote);
	    // -----------------------
		
		// Print header and instances: output of zero
		System.out.println("state_train\n");
		System.out.println(state_train);  // Outputs data, attribute, relation
		
		return state_train;
	}
	
	
	public static MultilayerPerceptron train_qlearning_nns(Instances state_train) throws Exception {
		MultilayerPerceptron q_network = new MultilayerPerceptron();
		//Setting Parameters
		q_network.setLearningRate(ALPHA);
		q_network.setMomentum(0.2);
		q_network.setTrainingTime(100); // number of training epochs
		q_network.setHiddenLayers("10,5,2"); // make a 2-layer NN, with binary output
	    
		q_network.buildClassifier(state_train);
		
		return q_network;
	}


	public static void evaluate_model(MultilayerPerceptron model) throws Exception {
		
		Evaluation evaluation = new Evaluation(state_train);
		evaluation.evaluateModel(model, state_train); // the output is an object of various evaluation metrics
		System.out.println(evaluation.toSummaryString());
		
		// For Initially trained model we just need to know the accuracy- (prediction evaluations across all samples of train data)
		// We do not need to output the prediction evaluations
		double accuracy = evaluation.pctCorrect();
		System.out.println("Accuracy = " + accuracy);
		
		// Best way to predict new instances
	    Instance state_instance = state_train.firstInstance();  // Take one instance, WORKS!!
	    
	    double[] prediction_prob = model.distributionForInstance(state_instance);
	    System.out.println("prediction_prob.length = " + prediction_prob.length);
	    System.out.println("prediction_prob = ");
	    for (int i=0;i<prediction_prob.length;i++){
	        System.out.println(prediction_prob[i] + " ");
	    }
	    
	    // Test classifyInstance
	    double result = model.classifyInstance(state_instance);
	    System.out.println("classifyInstance = ");
	    System.out.println(result);
	    
	}
}